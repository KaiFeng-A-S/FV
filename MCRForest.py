from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import parallel_helper
import numpy as np

def Encode(Forest, X):
    if not isinstance(Forest, RandomForestClassifier):
        raise TypeError('Not a legal Random Forest')
    
    X = Forest._validate_X_predict(X)
    results = Parallel(
        n_jobs = Forest.n_jobs, verbose = Forest.verbose, backend = 'threading')(
            delayed(parallel_helper)(tree, 'apply', X, check_input = False)
            for tree in Forest.estimators_)
    X_encode = np.array(results).T
    
    return X_encode

def calculate_parent(Tree):
    parent = np.full(Tree.node_count, -1, dtype = np.int32)
    for i in range(Tree.node_count):
        parent_from_left = np.where(Tree.children_left == i)[0]
        parent_from_right = np.where(Tree.children_right == i)[0]
        if len(parent_from_left) == 1:
            parent[i] = parent_from_left[0]
        elif len(parent_from_right) == 1:
            parent[i] = parent_from_right[0]
    
    return parent

def get_path_by_leaf(Tree, leaf_id):
#     print(Tree.tree_.node_count)
#     print(Tree.tree_.children_left)
#     print(Tree.tree_.children_right)
#     parent = Tree.tree_.parent
    parent = calculate_parent(Tree.tree_)
    path = []
    current_id = leaf_id
    while current_id > 0:
        path.insert(0, current_id)
        current_id = parent[current_id]
    path.insert(0, current_id)
    
    return path

def get_paths_by_leaves(Forest, leaves):
    paths = []
    for i , _tree_ in enumerate(Forest.estimators_):
        paths.append(get_path_by_leaf(_tree_, leaves[i]))
    
    return paths

def get_space_by_path(Tree, path, n_dims):
    space = np.full((n_dims, 2), np.nan, dtype = np.float32)
    for i, nid in enumerate(path[: -1]):
        parent = path[i]
        son = path[i + 1]
        tree_children_left = Tree.tree_.children_left
        tree_feature = Tree.tree_.feature
        tree_threshold = Tree.tree_.threshold
        is_left = tree_children_left[parent] == son
        k = tree_feature[parent]
        threshold = tree_threshold[parent]
        if is_left:
            space[k][1] = threshold
        else:
            space[k][0] = threshold
        
    return space

def get_spaces_by_leaves(Forest, leaves):
    paths = get_paths_by_leaves(Forest, leaves)
    spaces = []
    for i, _tree_ in enumerate(Forest.estimators_):
        space = get_space_by_path(_tree_, paths[i], Forest.n_features_)
        spaces.append(space)
    
    return spaces

def get_MCR_by_spaces(spaces, null_value = 0.):
    n_dims = len(spaces[0])
    _MCR_ = [[] for k in range(n_dims)]
    spaces = np.asarray(spaces)
    for k in range(n_dims):
        nan_left_mask = np.isnan(spaces[:, k, 0])
        nan_right_mask = np.isnan(spaces[:, k, 1])
        valid_left_mask = np.logical_not(nan_left_mask)
        valid_right_mask = np.logical_not(nan_right_mask)
        valid_left_index = np.where(valid_left_mask)[0]
        valid_right_index = np.where(valid_right_mask)[0]
        _MCR_[k].append(spaces[valid_left_index, k, 0])
        _MCR_[k].append(spaces[valid_right_index, k, 1])
    for i in range(n_dims):
        space = []
        if len(_MCR_[i][0]) == 0:
            space.append(null_value)
        else:
            space.append(np.max(_MCR_[i][0]))
        if len(_MCR_[i][1]) == 0:
            space.append(null_value)
        else:
            space.append(np.min(_MCR_[i][1]))
        _MCR_[i] = space
    
    return np.asarray(_MCR_)

def MCR(Forest, X_encode, null_value = 0.):
    X_MCR = []
    for single_encode in X_encode:
        spaces = get_spaces_by_leaves(Forest, single_encode)
        single_MCR = get_MCR_by_spaces(spaces, null_value)
        X_MCR.append(single_MCR)
    
    return np.asarray(X_MCR)
        

if __name__ == '__main__':
    X, y = make_classification(n_samples = 1000, n_features = 4, 
                               n_informative = 2, n_redundant = 0, 
                               random_state = 0, shuffle = False)

    clf = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 0)
    clf.fit(X, y)

#     first_tree = clf.estimators_[0]
    
#     print(isinstance(clf, RandomForestClassifier))
#     raise TypeError('Not a legal Random Forest')

#     print(Encode(clf, X))
#     print(clf.estimators_[0].tree_)
    
#     _, ptr = clf.decision_path(X)
#     print(_)
#     print(ptr)

    print(X)
    
#     from sklearn.tree._tree import Tree
#     for a in Tree.__dict__:
#         print(a)
    
    X_encode = Encode(clf, X)
    X_MCR = MCR(clf, X_encode)
    print(X_MCR[0])
    
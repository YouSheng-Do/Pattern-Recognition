"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np
import torch


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.feature_importance = np.zeros(X.shape[1])
        self.tree = self._grow_tree(X, y)
        self.feature_importance /= np.sum(self.feature_importance)

    def _grow_tree(self, X, y, depth=0):
        if depth == 0:
            self.num_samples = len(y)
        if depth != self.max_depth and len(np.unique(y)) != 1:
            current_gini_index = len(y) / self.num_samples * gini_index(y)
            # print(f"current:{current_gini_index}")
            feature_index, threshold = find_best_split(X, y)
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
            left_gini_index = len(left_y) / self.num_samples * gini_index(left_y)
            right_gini_index = len(right_y) / self.num_samples * gini_index(right_y)
            node_importance = current_gini_index - left_gini_index - right_gini_index
            # print(f"left:{left_gini_index}")
            # print(f"right:{right_gini_index}")
            self.feature_importance[feature_index] += node_importance
            leftchild = self._grow_tree(left_X, left_y, depth+1)
            rightchild = self._grow_tree(right_X, right_y, depth+1)
            return Node(
                        feature_index=feature_index,
                        left_child=leftchild,
                        right_child=rightchild,
                        threshold=threshold
                        )
        else:
            label = (np.bincount(y)).argmax()
            return Node(label=label)

    def predict(self, X):
        pred_y = []
        for x in X:
            pred_y.append(self._predict_tree(x, self.tree))
        return np.array(pred_y)

    def _predict_tree(self, x, tree_node):
        if tree_node.label != None:
            return tree_node.label
        if x[tree_node.feature_index] < tree_node.threshold:
            return self._predict_tree(x, tree_node.left_child)
        else:
            return self._predict_tree(x, tree_node.right_child)        

class Node:
    def __init__(self, left_child=None, right_child=None, feature_index=None, threshold=None, label=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature_index = feature_index
        self.threshold = threshold
        self.label = label

# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    left_X = X[X[:, feature_index] < threshold]
    left_y = y[X[:, feature_index] < threshold]
    right_X = X[X[:, feature_index] >= threshold]
    right_y = y[X[:, feature_index] >= threshold]

    return left_X, left_y, right_X, right_y

# Find the best split for the dataset
def find_best_split(X, y):
    min_entropy = float("inf")
    for feature_index in range(X.shape[1]):
        values = np.unique(X[:, feature_index])
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        for threshold in thresholds:
            _, left_y, _, right_y = split_dataset(X, y, feature_index, threshold)
            current_entropy = len(left_y) / len(y) * entropy(left_y) + \
                            len(right_y) / len(y) * entropy(right_y)
            if current_entropy < min_entropy:
                min_entropy = current_entropy
                best_feature_index = feature_index
                best_threshold = threshold
    
    return best_feature_index, best_threshold

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    return - np.sum(counts / len(y) * np.log2(counts / len(y)))

def gini_index(y):
    _, counts = np.unique(y, return_counts=True)
    probability = counts / len(y)
    return 1 - np.sum(probability ** 2)
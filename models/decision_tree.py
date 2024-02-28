import numpy as np

"""
Steps for building a decision tree

  - Start with all examples at the root node
  - Calculate information gain for splitting on all possible features,
    and pick the one with the highest information gain
  - Split dataset according to the selected feature, 
    and create left and right branches of the tree
  - Keep repeating splitting process until stopping criteria is met

"""


class DecisionTree(object):
    def __init__(self):
        self.tree = []

    def compute_entropy(self, y):

        entropy = 0

        if len(y) == 0:
            return 0
        p1 = sum(y[y == 1]) / len(y)
        if p1 == 0 or p1 == 1:
            return 0
        else:

            return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    def information_gain(self, X, y, node_indices, feature):

        left_indices, right_indices = self.split_dataset(X, node_indices, feature)

        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        info_gain = 0

        node_entropy = self.compute_entropy(y_node)
        left_entropy = self.compute_entropy(y_node)
        right_entropy = self.compute_entropy(y_node)

        w_left = len(X_left) / len(X_node)
        w_right = len(X_right) / len(X_node)

        weighted_entropy = w_left * left_entropy + w_right * right_entropy
        info_gain = node_entropy - weighted_entropy

        return info_gain

    def split_dataset(self, X, node_indices, feature):

        left_indices = []
        right_indices = []

        for i in node_indices:
            if X[i][feature] == 1:
                left_indices.append(i)
            else:
                right_indices.append(i)

        return left_indices, right_indices

    def best_split(self, X, y, node_indices):
        num_features = X.shape[1]

        best_feature = -1

        max_info_gain = 0
        for feature in range(num_features):
            info_gain = self.information_gain(X, y, node_indices, feature)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        return best_feature

    def build_tree(self, X, y, node_indices, branch_name, max_depth, current_depth):

        if current_depth == max_depth:
            formatting = " " * current_depth + "-" * current_depth
            print(formatting, "%s leaf node with indices" % branch_name, node_indices)
            return

        best_feature = self.best_split(X, y, node_indices)

        formatting = "-" * current_depth
        print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

        left_indices, right_indices = self.split_dataset(X, node_indices, best_feature)
        self.tree.append((left_indices, right_indices, best_feature))

        self.build_tree(X, y, left_indices, "Left", max_depth, current_depth + 1)
        self.build_tree(X, y, right_indices, "Right", max_depth, current_depth + 1)

        return self.tree

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree[0]
            while True:
                left_indices, right_indices, feature = node
                if feature == -1:
                    predictions.append(left_indices[0])
                    break
                if sample[feature] == 1:
                    if left_indices:
                        node = self.tree[left_indices[0]]
                    else:
                        predictions.append(1)
                        break
                else:
                    if right_indices:
                        node = self.tree[right_indices[0]]
                    else:
                        predictions.append(0)
                        break
        return predictions


X_train = np.array([[1, 1, 1],
 [0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

Y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

model = DecisionTree()
tree = model.build_tree(X_train, Y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=2, current_depth=0)
prediction = model.predict(X_train)

print(model.tree)
print(Y_train)
print(prediction)





from DecisionTree.Common import *
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

def variance_error(data_set):
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]

class CART:
    def __init__(self):
        self.__prune_tree = dict()
        self.__tree: dict = dict()
        self.__data_matrix: np.ndarray = np.empty((0, 0))
        self.__test_matrix: np.ndarray = np.empty((0, 0))
        self.__ops = (1, 4)

    def load_data(self, data_matrix: np.ndarray, test_data: np.ndarray):
        self.__data_matrix = data_matrix
        self.__test_matrix = test_data

    @property
    def tree(self):
        return self.__tree

    @classmethod
    def reg_leaf(cls, data_set):
        return np.mean(data_set[:, -1])

    @classmethod
    def split_data_set(cls, data_set, feature, value):
        mat0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
        mat1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
        return mat0, mat1

    @classmethod
    def is_tree(cls, obj):
        return type(obj).__name__ == 'dict'

    @classmethod
    def get_mean(cls, tree):
        if cls().is_tree(tree['right']):
            tree['right'] = cls().get_mean(tree['right'])
        if cls().is_tree(tree['left']):
            tree['left'] = cls().get_mean(tree['left'])

        return (tree['left'] + tree['right']) / 2.0

    def __choose_best_feature(self, data_set: np.ndarray):
        tolS, tolN = self.__ops
        # if all the target variables are the same value: quit and return value
        if len(set(data_set[:, -1].transpose().tolist())) == 1:
            return None, CART.reg_leaf(data_set)

        m, n = np.shape(data_set)
        # the choice of the best feature is driven by Reduction in RSS error from mean
        S = variance_error(data_set)
        best_s = np.inf; best_index = 0; best_value = 0
        for feat_index in range(n - 1):
            for split_val in set(data_set[:, feat_index]):
                mat0, mat1 = CART.split_data_set(data_set, feat_index, split_val)
                if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                    continue

                new_s = variance_error(mat0) + variance_error(mat1)
                if new_s < best_s:
                    best_index = feat_index
                    best_value = split_val
                    best_s = new_s

        # if the decrease (S-bestS) is less than a threshold don't do the split
        # 下面都是异常处理
        if (S - best_s) < tolS:
            return None, CART.reg_leaf(data_set)  # exit cond 2
        mat0, mat1 = CART.split_data_set(data_set, best_index, best_value)
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit cond 3
            return None, CART.reg_leaf(data_set)

        return best_index, best_value  # returns the best feature to split on
        # and the value used for that split

    @classmethod
    def __prune(cls, tree, test_data):
        if np.shape(test_data)[0] == 0:
            return CART.get_mean(tree)  # if we have no test data collapse the tree

        has_left_tree = CART.is_tree(tree['left'])
        has_right_tree = CART.is_tree(tree['right'])

        l_set, r_set = None, None
        if has_left_tree or has_right_tree:  # if the branches are not trees try to prune them
            l_set, r_set = CART.split_data_set(test_data, tree['spInd'], tree['spVal'])

        if has_left_tree:
            tree['left'] = CART.__prune(tree['left'], l_set)
        if has_right_tree:
            tree['right'] = CART.__prune(tree['right'], r_set)

        # if they are now both leafs, see if we can merge them
        if not cls().is_tree(tree['left']) and not cls().is_tree(tree['right']):
            l_set, r_set = CART.split_data_set(test_data, tree['spInd'], tree['spVal'])
            error_no_merge = sum(np.power(l_set[:, -1] - tree['left'], 2)) + \
                           sum(np.power(r_set[:, -1] - tree['right'], 2))
            tree_mean = (tree['left'] + tree['right']) / 2.0
            error_merge = sum(np.power(test_data[:, -1] - tree_mean, 2))
            if error_merge < error_no_merge:
                return tree_mean
            else:
                return tree
        else:
            return tree

    def __create_tree(self, data_set):
        best_index, best_value = self.__choose_best_feature(data_set)
        if best_index is None:
            return best_value  # if the splitting hit a stop condition return val

        ret_tree = dict()
        
        ret_tree['spInd'] = best_index
        ret_tree['spVal'] = best_value
        l_set, r_set = CART.split_data_set(data_set, best_index, best_value)
        ret_tree['left'] = self.__create_tree(l_set)
        ret_tree['right'] = self.__create_tree(r_set)
        return ret_tree

    def create_tree(self):
        self.__tree = self.__create_tree(self.__data_matrix)
        self.__prune_tree = self.__prune(self.tree.copy(), self.__test_matrix)
        pass


def main():
    # data_set = load_iris()
    # train_labels = data_set['target']
    # train_labels[train_labels != 0] = 1
    # train_data = data_set['data']

    train_data = np.array(load_data_set2("DecisionTree//data//bikeSpeedVsIq_train.txt"))
    test_data = np.array(load_data_set2("DecisionTree//data//bikeSpeedVsIq_test.txt"))

    cart = CART()
    cart.load_data(train_data, test_data)
    cart.create_tree()
    ret = cart.tree

    plt.scatter(train_data[:, 1], train_data[:, 0])
    plt.show()

if __name__ == "__main__":
    main()


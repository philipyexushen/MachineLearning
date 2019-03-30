from DecisionTree.Common import *

def variance_error(data_set):
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]

class CART:
    def __init__(self):
        self.__data_matrix: np.ndarray = None
        self.__ops = (1, 4)

    def load_data(self, data_matrix: np.ndarray):
        self.__data_matrix = data_matrix

    @classmethod
    def reg_leaf(cls, data_set):
        return np.mean(data_set[:, -1])

    @classmethod
    def split_data_set(cls, data_set, feature, value):
        mat0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
        mat1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
        return mat0, mat1

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
            fuck = set(data_set[:, feat_index])
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
        return self.__create_tree(self.__data_matrix)

def main():
    train_data = load_data_set2("DecisionTree//data//ex0.txt")
    cart = CART()
    cart.load_data(np.array(train_data))
    ret = cart.create_tree()
    print(ret)

if __name__ == "__main__":
    main()


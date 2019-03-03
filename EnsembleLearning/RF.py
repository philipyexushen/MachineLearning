import numpy as np
from EnsembleLearning.DecisionStump import *

class RandomForest:
    def __init__(self):
        self.__data_matrix: np.matrix = None
        self.__class_labels: np.matrix = None
        self.__tree_set = None

    def load_data(self, data_matrix: np.matrix, class_labels: np.matrix):
        self.__data_matrix: np.matrix = data_matrix
        self.__class_labels: np.matrix = class_labels
        self.__tree_set = None

    @property
    def data_matrix(self):
        return self.__data_matrix

    @property
    def class_labels(self):
        return self.__class_labels

    def __generate_new_data_set(self):
        m, n = np.shape(self.data_matrix)
        selected_index = np.random.choice(m, int(m / 2))
        return self.__data_matrix[selected_index], self.class_labels[:, selected_index]

    def train(self, forest_num = 100):
        tree_set = []

        for _ in range(forest_num):
            new_data_set, new_class_labels = self.__generate_new_data_set()
            m, n = np.shape(new_data_set)
            D = np.mat(np.ones((m, 1)) / m)

            selected_index = np.random.choice(n, int(n /2))

            stump = DecisionStump()
            stump.load_data(new_data_set, new_class_labels, D)
            best_stump, *_ = stump.buildStump(selected_index)

            tree_set.append(best_stump)

        self.__tree_set = tree_set

    def test(self, test_data:np.matrix)->np.ndarray:
        m, n = np.shape(test_data)
        result = np.zeros((m, 1), dtype=np.float)
        for tree in self.__tree_set:
            class_est = DecisionStump.stumpClassify(test_data, tree['dim'], tree['thresh'], tree['ineq'])
            result += class_est

        return np.sign(result)



















    


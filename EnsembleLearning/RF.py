import numpy as np

class RandomForest:
    def __init__(self):
        self.__data_matrix: np.matrix = None
        self.__class_labels: np.matrix = None

    def load_data(self, data_matrix: np.matrix, class_labels: np.matrix):
        self.__data_matrix: np.matrix = data_matrix
        self.__class_labels: np.matrix = class_labels

    @property
    def data_matrix(self):
        return self.__data_matrix

    @property
    def class_labels(self):
        return self.__class_labels

    def __generate_new_data_set(self):
        m, n = np.shape(self.data_matrix)
        return self.__data_matrix[np.random.choice(m, m)]

    def train(self, iteration = 500):
        tree_set = []

        for _ in range(iteration):
            new_data_set = self.__generate_new_data_set()

    


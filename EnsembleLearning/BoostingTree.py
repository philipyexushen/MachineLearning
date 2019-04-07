import numpy as np

class BoostingTree:
    def __init__(self):
        self.__data_matrix: np.matrix = None
        self.__class_labels: np.matrix = None
        self.__weak_class_arr = None

    def load_data(self, data_matrix: np.matrix, class_labels: np.matrix):
        self.__data_matrix = data_matrix
        self.__class_labels = class_labels

    @property
    def data_matrix(self):
        return self.__data_matrix

    @property
    def class_labels(self):
        return self.__class_labels

    def train(self):
        # TODO
        pass
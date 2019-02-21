import numpy as np

class DecisionStump:
    def __init__(self):
        self.__data_matrix:np.matrix = None
        self.__class_labels:np.matrix = None
        self.__init_cof:np.matrix = None

    def load_data(self, data_matrix:np.matrix, class_labels:np.matrix, init_cof:np.matrix):
        self.__data_matrix:np.matrix = data_matrix
        self.__class_labels:np.matrix = class_labels
        self.__init_cof:np.matrix = init_cof

    @classmethod
    def stumpClassify(cls, data_matrix, dim, thresh_value, thresh_inequality_label):
        retArray = np.ones((np.shape(data_matrix)[0], 1))
        if thresh_inequality_label == 'lt':
            retArray[data_matrix[:, dim] <= thresh_value] = -1.0
        else:
            retArray[data_matrix[:, dim] > thresh_value] = -1.0
        return retArray

    def buildStump(self):
        data_matrix = self.__data_matrix; label_mat = self.__class_labels.T
        m, n = np.shape(data_matrix)
        num_steps = 10.0
        best_stump = {}
        best_class_est = np.mat(np.zeros((m, 1)))
        min_error = np.inf
        for i in range(n):
            range_min = data_matrix[:, i].min(); range_max = data_matrix[:, i].max()
            step_size = (range_max - range_min) / num_steps

            for j in range(-1, int(num_steps) + 1):
                for inequality_label in ['lt', 'gt']:
                    thresh_val = (range_min + float(j) * step_size)
                    predicted_val = DecisionStump.stumpClassify(self.__data_matrix, i, thresh_val, inequality_label)

                    errArr = np.mat(np.ones((m, 1)))
                    errArr[predicted_val == label_mat] = 0
                    weighted_error = self.__init_cof.T * errArr

                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_class_est = predicted_val.copy()
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequality_label
        return best_stump, min_error, best_class_est

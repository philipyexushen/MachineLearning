from EnsembleLearning.DecisionStump import *

class AdaBoostClassifier:
    def __init__(self):
        self.__data_matrix:np.matrix = None
        self.__class_labels:np.matrix = None
        self.__weak_class_arr = None

    def load_data(self, data_matrix:np.matrix, class_labels:np.matrix):
        self.__data_matrix = data_matrix
        self.__class_labels = class_labels

    @property
    def data_matrix(self):
        return self.__data_matrix

    @property
    def class_labels(self):
        return self.__class_labels

    def train(self, max_iter = 100):
        m, n = np.shape(self.__data_matrix)
        D = np.mat(np.ones((m, 1)) / m)
        agg_class_est = np.mat(np.zeros((m, 1)))
        self.__weak_class_arr = []

        for i in range(max_iter):
            stump = DecisionStump()
            stump.load_data(self.data_matrix, self.class_labels, D)
            best_stump, error, class_est = stump.buildStump()

            # 计算Gm(x)的系数 am = 1/2 log(1- em/ em) em为误差，在这里为单决策树的错误率
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
            best_stump['alpha'] = alpha
            self.__weak_class_arr.append(best_stump)

            # 计算新的D
            D = np.multiply(D, np.exp(np.multiply(-1 * alpha * np.mat(self.class_labels).T, class_est)))
            # 规范化
            D = D / D.sum()

            # 计算f(x) 即基本分类器组合
            agg_class_est += alpha * class_est
            agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(self.class_labels).T, np.ones((m, 1)))
            error_rate = agg_errors.sum() / m
            if error_rate == 0.0:
                break

    def test(self, test_data:np.matrix)->np.ndarray:
        classifier = self.__weak_class_arr
        m, n = np.shape(self.data_matrix)
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(len(classifier)):
            classEst = DecisionStump.stumpClassify(test_data, classifier[i]['dim'], classifier[i]['thresh'], classifier[i]['ineq'])
            aggClassEst += classifier[i]['alpha'] * classEst
        return np.sign(aggClassEst)

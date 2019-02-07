import numpy as np
import random

def rbf_kernel(data_mat, alpha):
    m, n = np.shape(data_mat)
    kernel = np.mat(np.zeros((m, m)))
    for i in range(m):
        K = np.mat(np.zeros((m, 1)))
        row = data_mat[i, :]
        for j in range(m):
            deltaRow = data_mat[j, :] - row
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * alpha ** 2))  # divide in NumPy is element-wise not matrix like Matlab
        kernel[:,i] = K
    return kernel

class SMOPlatt:
    """
    带核函数版本的完整的SMO算法实现
    """

    def __init__(self, data_mat, label_mat, kernel_generator):
        self.__data_set: np.ndarray = data_mat
        self.__label_mat: np.ndarray = label_mat

        m, n = np.shape(data_mat)
        self.__alphas = np.mat(np.zeros((m, 1)))
        self.__eCache = np.mat(np.zeros((m, 2)))
        self.__b = 0

        # 用核函数映射一下
        self.__K = kernel_generator(np.mat(self.__data_set))

    @property
    def alphas(self): return self.__alphas

    @property
    def b(self): return self.__b

    @staticmethod
    def __clip_alpha(ai, H, L):
        ret = min(ai, H)
        ret = max(ret, L)
        return ret

    @staticmethod
    def __unique_random(i, data_range):
        j = i
        while j == i:
            j = int(random.uniform(0, data_range))
        return j

    def __unique_random_cache(self, index, Ei, alphas, label_mat, data_matrix, b):
        maxK = -1; maxDeltaE = 0; Ej = 0
        self.__eCache[index] = [1, Ej]  # set valid #choose the alpha that gives the maximum delta E
        m, n = np.shape(self.__data_set)

        valid_cache_list = np.nonzero(self.__eCache[:, 0].A)[0]
        if (len(valid_cache_list)) > 1:
            for k in valid_cache_list:  # loop through valid Ecache values and find the one that maximizes delta E
                if k == index: continue  # don't calc for i, waste of time
                Ek = self.__get_Ek(index, alphas, label_mat, data_matrix, b)

                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej

        else:  # in this case (first time around) we don't have any valid eCache values
            j = self.__unique_random(index, m)
            Ek = self.__get_Ek(index, alphas, label_mat, data_matrix, b)

        return j, Ek

    @staticmethod
    def __get_Ek(k, alphas, label_mat, data_matrix, b):
        fxi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[k, :].T)) + b
        Ek = fxi - float(label_mat[k])
        return Ek

    def __updateEk(self, k, alphas, label_mat, data_matrix, b):
        # after any alpha has changed update the new value in the cache
        Ek = self.__get_Ek(k, alphas, label_mat, data_matrix, self.__b)
        self.__eCache[k] = [1, Ek]


    def apply(self, C, toler = 1e-3, max_iter = 10000):
        entireSet = True
        alphaPairsChanged:bool = False
        it = 0

        m, n = np.shape(self.__data_set)
        while (it < max_iter) and ((alphaPairsChanged == True) or entireSet):
            alphaPairsChanged = False
            if entireSet:  # go over all
                for i in range(m):
                    ret = self.__inner_apply(i, C, toler, max_iter)
                    alphaPairsChanged = ret or alphaPairsChanged
                it += 1
            else:  # go over non-bound (railed) alphas
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < C))[0]
                for i in nonBoundIs:
                    ret = self.__inner_apply(i, C, toler, max_iter)
                    alphaPairsChanged = ret or alphaPairsChanged
                it += 1
            if entireSet:
                entireSet = False  # toggle entire set loop
            elif not alphaPairsChanged:
                entireSet = True


    def __inner_apply(self, i, C, toler, max_iter):
        data_matrix = np.mat(self.__data_set)
        label_mat = np.mat(self.__label_mat).transpose()
        alphas = self.alphas
        m, n = np.shape(data_matrix)
        Ei = self.__get_Ek(i, alphas, label_mat, data_matrix, self.__b)

        # Ei是误差，所以下面的意思是，误差越大，越值得优化，并且alpha已经保证了不能在边界上(0和C)了（看下面的代码的clip）
        # magic代码，可以加快迭代速度，但是不知道为什么
        if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
            j, Ej = self.__unique_random_cache(i, Ei, alphas, label_mat, data_matrix, self.__b)

            alpha_i_old = alphas[i].copy(); alpha_j_old = alphas[j].copy()

            # 方形约束
            if label_mat[i] != label_mat[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
            else:
                L = max(0, -C + alphas[j] + alphas[i])
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                return False

            # eta = -(K11 + K22 - 2K12)
            eta = -data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T \
                  + 2 * data_matrix[i, :] * data_matrix[j, :].T

            # a2new = a2Old + yj * (E1 - E2) / eta ==> update a2
            if eta == 0:
                return False

            alphas[j] -= label_mat[j] * (Ei - Ej) / eta
            alphas[j] = self.__clip_alpha(alphas[j], H, L)

            # 缓存一次Ej
            self.__updateEk(j, alphas, label_mat, data_matrix, self.__b)

            # 更新量太少了，直接忽略好了
            if abs(alphas[j] - alpha_j_old) < 1e-6:
                # print("j not moving enough")
                return False

            # a1new = a1Old + yi* yj * (a2Old - a2new) / eta ==> update a1
            alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

            # 缓存一次Ei
            self.__updateEk(i, alphas, label_mat, data_matrix, self.__b)

            # 通过a更新b
            b1 = self.__b - Ei - float(
                label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T) - \
                 float(label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T)
            b2 = self.__b - Ej - float(
                label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T) - \
                 float(label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T)

            if 0 < alphas[i] < C:
                self.__b = b1
            elif 0 < alphas[j] < C:
                self.__b = b2
            else:
                self.__b = (b1 + b2) / 2.0
        return True
import numpy as np
import random

class SMOSimple:
    def __init__(self, data_mat, label_mat):
        self.__data_set:np.ndarray = data_mat
        self.__label_mat:np.ndarray = label_mat

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

    def apply(self, C, toler = 1e-3, max_iter = 10000):
        data_matrix = np.mat(self.__data_set); label_mat = np.mat(self.__label_mat).transpose()
        b = 0
        m, n = np.shape(data_matrix)
        alphas = np.mat(np.zeros((m, 1)))

        it = 0
        while it < max_iter:
            alphaPairsChanged:bool = False
            for i in range(m):
                # Ei = f(xi) - yi
                # f(xi) = sum(w* x + b)
                fxi = float(np.multiply(alphas, label_mat).T * (data_matrix*data_matrix[i,:].T)) + b
                Ei = fxi - float(label_mat[i])

                # Ei是误差，所以下面的意思是，误差越大，越值得优化，并且alpha已经保证了不能在边界上(0和C)了（看下面的代码的clip）
                # magic代码，可以加快迭代速度，但是不知道为什么
                if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                    j = self.__unique_random(i, m)
                    fxj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                    Ej = fxj - float(label_mat[j])
                    alpha_i_old = alphas[i].copy(); alpha_j_old = alphas[j].copy()

                    # 方形约束
                    if label_mat[i] != label_mat[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, -C + alphas[j] + alphas[i])
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        continue
                    # eta = -(K11 + K22 - 2K12)
                    eta = -data_matrix[i,:] * data_matrix[i,:].T - data_matrix[j,:] * data_matrix[j,:].T \
                        + 2 * data_matrix[i,:] * data_matrix[j,:].T

                    # a2new = a2Old + yj * (E1 - E2) / eta ==> update a2
                    if eta == 0:
                        continue

                    alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                    alphas[j] = self.__clip_alpha(alphas[j], H, L)

                    # 更新量太少了，直接忽略好了
                    if abs(alphas[j] - alpha_j_old) < 1e-6:
                        # print("j not moving enough")
                        continue

                    # a1new = a1Old + yi* yj * (a2Old - a2new) / eta ==> update a1
                    alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                    # 通过a更新b
                    b1 = b - Ei - float(label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T) - \
                         float(label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T)
                    b2 = b - Ej - float(label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T) - \
                         float(label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T)

                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged = True
            it = it + 1 if not alphaPairsChanged else 0

        # alphas中大于零的样本点(xi, yi)的实例xi称为支持向量，或者软间隔支持向量
        return b, alphas










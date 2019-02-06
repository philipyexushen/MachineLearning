import numpy as np
import logging as log
import random

def unique_random(i, data_range):
    j = i
    while j==i:
        j = int(random.uniform(0, data_range))
    return j

class SMOSimple:
    def __init__(self, data_mat, label_mat):
        self.__data_set:np.ndarray = data_mat
        self.__label_mat:np.ndarray = label_mat

    @staticmethod
    def __clip_alpha(ai, H, L):
        ret = min(ai, H)
        ret = max(ret, L)
        return ret

    def apply(self, C, toler = 0.001, max_iter = 10000):
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
                fxi = np.multiply(alphas, label_mat).T * (data_matrix*data_matrix[i,:].T) + b
                Ei = fxi - label_mat[i]

                # 下面这个是什么约束？toler是个what
                if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                    j = unique_random(i, m)
                    fxj = np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T) + b
                    Ej = fxj - label_mat[j]
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
                    alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                    alphas[j] = self.__clip_alpha(alphas[j], H, L)

                    # 更新量太少了，直接忽略好了
                    if abs(alphas[j] - alpha_j_old) < 0.00001:
                        # print("j not moving enough")
                        continue

                    # a1new = a1Old + yi* yj * (a2Old - a2new) / eta ==> update a1
                    alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                    # 通过a更新b
                    b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                         label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                    b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                         label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T

                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged = True
            it = it + 1 if not alphaPairsChanged else 0

        return b, alphas










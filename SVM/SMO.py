import numpy as np
from matplotlib import pyplot as plt
from SVM.SMOPlatt import *
from SVM.SMOSimple import *

def load_data_set(fileName):
    dataMat = []; labelMat = []

    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))

    return dataMat,labelMat

class DrawSVMDataHelper:
    @staticmethod
    def __get_coefficient(alphas, data_mat, label_mat):
        alphas, data_mat, labels = np.array(alphas), np.array(data_mat), np.array(label_mat)
        yx = labels.reshape(1, -1).T * np.ones((1, np.shape(data_mat)[1])) * data_mat
        w = np.dot(yx.T, alphas)
        return w.tolist()

    @classmethod
    def draw(cls, data_mat, label_mat, alphas, b, is_draw_cutoff_line:bool = True):
        classified_pts = {'+1': [], '-1': []}
        for point, label in zip(data_mat, label_mat):
            if label == 1.0:
                classified_pts['+1'].append(point)
            else:
                classified_pts['-1'].append(point)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, pts in classified_pts.items():
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], label=label)

        if is_draw_cutoff_line:
            w = cls().__get_coefficient(alphas, data_mat, label_mat)
            x1, _ = max(data_mat, key=lambda x: x[0])
            x2, _ = min(data_mat, key=lambda x: x[0])
            a1, a2 = w
            y1, y2 = (-b - float(a1[0]) * x1) /  float(a2[0]), (-b -  float(a1[0]) * x2) /  float(a2[0])
            ax.plot([x1, x2], [y1, y2])

        # 绘制支持向量, 去看一下支持向量的定义
        sum_svm_support = 0
        for i, alpha in enumerate(alphas):
            if abs(alpha) > 1e-3:
                x, y = data_mat[i]
                ax.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='#AB3319')
                sum_svm_support += 1

        print(f"sum_svm_support={sum_svm_support}")
        plt.show()

def main():
    # data_mat, label_mat = load_data_set("SVMData/testSet.txt")
    # SMOFactory = SMOSimple(data_mat, label_mat)
    # b, alphas = SMOFactory.apply(0.6)

    data_mat, label_mat = load_data_set("SVMData/testSetRBF.txt")
    SMOFactory = SMOPlatt(data_mat, label_mat, lambda data_set : rbf_kernel(data_set, 0.1))
    SMOFactory.apply(0.68)
    alphas = SMOFactory.alphas
    b = SMOFactory.b

    check_data_mat, check_label_mat = load_data_set("SVMData/testSetRBF2.txt")
    correct_rate = SMOFactory.check(check_data_mat, check_label_mat)
    print(f"correct_rate={correct_rate}")
    DrawSVMDataHelper.draw(data_mat, label_mat, alphas, b, False)

if __name__ == "__main__":
    main()
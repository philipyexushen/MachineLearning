import numpy as np
from SMOSimple import *

def load_data_set(fileName):
    dataMat = []; labelMat = []

    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))

    return dataMat,labelMat


def main():
    data_mat, label_mat = load_data_set("SVMData/testSet.txt")
    SMOFactory = SMOSimple(data_mat, label_mat)
    SMOFactory.apply(0.6)

if __name__ == "__main__":
    main()
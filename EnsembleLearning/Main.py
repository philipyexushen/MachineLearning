from EnsembleLearning.AdaBoosting import *
from EnsembleLearning.DecisionStump import DecisionStump

def load_data_set(file_name):
    numFeat = len(open(file_name).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []

    with open(file_name) as fr:
        for line in fr.readlines():
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))

    return dataMat,labelMat

def main():
    train_data, train_class_labels = load_data_set("EnsembleLearning\\horseColicTraining2.txt")
    classifier = AdaBoostClassifier()
    classifier.load_data(np.matrix(train_data), np.matrix(train_class_labels))
    classifier.train(100)

    test_data, test_class_labels = load_data_set("EnsembleLearning\\horseColicTraining2.txt")
    prediction = classifier.test(np.matrix(test_data), np.matrix(test_class_labels))

    test_data_size = len(test_data)
    err = np.mat(np.ones((test_data_size, 1)))
    print(err[prediction != np.mat(test_class_labels).T].sum() / test_data_size)

if __name__ == "__main__":
    main()
from EnsembleLearning.AdaBoosting import *
from EnsembleLearning.RF import *

def load_data_set(file_name):
    num_feat = len(open(file_name).readline().split('\t')) #get number of fields
    data_mat = []; label_mat = []

    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr =[]
            cur_line = line.strip().split('\t')
            for i in range(num_feat-1):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))

    return data_mat,label_mat

def main():
    train_data, train_class_labels = load_data_set("EnsembleLearning\\horseColicTraining2.txt")
    test_data, test_class_labels = load_data_set("EnsembleLearning\\horseColicTraining2.txt")
    test_data_size = len(test_data)

    print("train Random Forest")
    RF = RandomForest()
    RF.load_data(np.matrix(train_data), np.matrix(train_class_labels))
    RF.train(10000)
    prediction = RF.test(np.matrix(test_data))
    err = np.mat(np.ones((test_data_size, 1)))
    print(err[prediction != np.mat(test_class_labels).T].sum() / test_data_size)

    print("train Adaboost")
    classifier = AdaBoostClassifier()
    classifier.load_data(np.matrix(train_data), np.matrix(train_class_labels))
    classifier.train(100)
    prediction = classifier.test(np.matrix(test_data))
    err = np.mat(np.ones((test_data_size, 1)))
    print(err[prediction != np.mat(test_class_labels).T].sum() / test_data_size)

if __name__ == "__main__":
    main()
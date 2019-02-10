import numpy as np

class DocumentClassifier:
    def __init__(self):
        self.__doc_list = None
        self.__class_list = None
        self.__vocab_set = None
        self.__p0V = None
        self.__p1V = None
        self.__pAb = None

    def load_data_set(self, doc_list, class_list)->None:
        self.__doc_list = doc_list
        self.__class_list = class_list

        vocab_set = set([])
        for document in self.__doc_list:
            vocab_set = vocab_set | set(document)
        self.__vocab_set = list(vocab_set)

    def train(self)->None:
        train_mat = []
        for doc in self.__doc_list:
            train_mat.append(self.__set_words_to_vec(self.__vocab_set, doc))
        self.__p0V, self.__p1V, self.__pAb = self.__trainNB0(np.array(train_mat), np.array(self.__class_list))

    def test(self, entry)->bool:
        this_doc = np.array(self.__set_words_to_vec(self.__vocab_set, entry))
        return self.__classifyNB(this_doc,  self.__p0V, self.__p1V, self.__pAb)

    @staticmethod
    def __trainNB0(train_matrix, train_category):
        num_train_docs, num_words = np.shape(train_matrix)
        p_abusive = sum(train_category) / float(num_train_docs)
        p0_num = np.ones(num_words); p1_num = np.ones(num_words)  # change to ones()
        p0_denominator = 2.0; p1_denominator = 2.0  # change to 2.0
        for i in range(num_train_docs):
            if train_category[i] == 1:
                p1_num += train_matrix[i]
                p1_denominator += sum(train_matrix[i])
            else:
                p0_num += train_matrix[i]
                p0_denominator += sum(train_matrix[i])
        p1_vec = np.log(p1_num / p1_denominator)
        p0_vec = np.log(p0_num / p0_denominator)
        # 得到各个词条件概率，以及先验概率pAbusive
        return p0_vec, p1_vec, p_abusive

    @staticmethod
    def __classifyNB(vec2Classify, p0Vec, p1Vec, pClass1)->bool:
        p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # element-wise mult
        p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
        return p1 > p0

    @staticmethod
    def __set_words_to_vec(vocab_list, input_set):
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                # 词袋模型
                return_vec[vocab_list.index(word)] += 1
        return return_vec


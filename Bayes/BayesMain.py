from Bayes.DocumentClassifier import *
import re
import random
import os

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def main():
    classifier = DocumentClassifier()
    '''
    classifier = DocumentClassifier()
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性词，0表示非侮辱性词

    classifier.load_data_set(posting_list, class_vec)
    classifier.train()

    ret = classifier.test(["stupid", "worthless", "dog"])
    print(f"result={ret}")
    '''

    doc_list = []; class_list = []; full_text = []
    for i in range(1, 26):
        with open(f"Bayes/email/spam/{i}.txt", errors="ignore") as fp_spam:
            wordList = textParse(fp_spam.read())
            doc_list.append(wordList)
            full_text.extend(wordList)
            class_list.append(1)

        with open(f"Bayes/email/ham/{i}.txt", errors="ignore") as fp_ham:
            wordList = textParse(fp_ham.read())
            doc_list.append(wordList)
            full_text.extend(wordList)
            class_list.append(0)

    classifier.load_data_set(doc_list[:30], class_list[:30])
    classifier.train()

    test_set = []
    already_select_list = []
    remain_select = list(range(0, len(doc_list)))
    for i in range(10):
        rand_index = int(random.uniform(0, len(remain_select)))
        test_set.append(doc_list[remain_select[rand_index]])
        already_select_list.append(remain_select[rand_index])
        del remain_select[rand_index]

    error_count = 0
    for i in range(len(test_set)):  # classify the remaining items
        doc_index = already_select_list[i]
        if classifier.test(test_set[i]) != class_list[doc_index]:
            error_count += 1

    print(f"error_rate={ error_count / len(test_set)}")


if __name__ == "__main__":
    main()
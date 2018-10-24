from Common import *
from DataManager import *
from matplotlib import pyplot as plt

class TreeNode:
    def __init__(self):
        self.__category = None
        self.__child = []

    @property
    def category(self):
        return self.__category

    @category.setter
    def category(self, value):
        self.__category = value

    @property
    def child(self):
        return self.__child


class DecisionTreeBuilder:
    def __init__(self):
        mgr = ExcelDataManager()
        self.__data_set = mgr.fetch()

    @property
    def data_set(self):
        return self.__data_set

    def __tree_generate(self, data_set, attribute_list, root:TreeNode):
        classified_list = self.__get_classified_list(data_set)
        # 看下数据集的元素是不是都是同一个类别的，如果都是同一个类别的，直接返回就好了
        if len(classified_list) == 1:
            root.category = classified_list[0]
            return



    @staticmethod
    def __get_classified_list(D):
        return [item[-1] for item in D]

    def tree_generate(self):
        classified_list = self.__get_classified_list(self.data_set)
        root = TreeNode()
        return self.__tree_generate(self.data_set, classified_list, root)


if __name__ == "__main__":
    builder = DecisionTreeBuilder()
    builder.tree_generate()
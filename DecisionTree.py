from Common import *
from DataManager import *
from matplotlib import pyplot as plt

class TreeNode:
    def __init__(self):
        self.__category = None
        self.__division_attribute = None
        self.__child = []

    @property
    def category(self):
        return self.__category

    @category.setter
    def category(self, value):
        self.__category = value

    @property
    def division_attribute(self):
        return self.__division_attribute

    @division_attribute.setter
    def division_attribute(self, value):
        self.__division_attribute = value

    @property
    def child(self):
        return self.__child


class DecisionTreeBuilder:
    def __init__(self):
        mgr = ExcelDataManager()
        self.__data_set = mgr.fetch()

        attribute_list = mgr.get_all_title()
        self.__attribute_list = attribute_list[1:len(attribute_list) - 1]

    @property
    def data_set(self):
        return self.__data_set

    def __tree_generate(self, data_set, root:TreeNode, division_attribute):
        """
        :param data_set: 数据集
        :param attribute_list: 属性集（比如西瓜数据集的色泽，根蒂，敲声，纹理等）
        :param root: 决策树当前根节点
        :return: 决策树当前根节点
        """
        classified_list = self.__get_classified_list(data_set)
        # 看下数据集的元素是不是都是同一个类别的，如果都是同一个类别的，直接返回就好了
        if len(set(classified_list)) == 1:
            root.division_attribute = division_attribute
            root.category = classified_list[0]
            return root
        # 如果数据集的各属性的取值都是一样的
        if self.__check_if_same_attribute(data_set):
            pass


    @staticmethod
    def __check_if_same_attribute(data_set):
        for index in range(len(data_set)):
            data_column = data_set[:, index]
            if len(set(data_column)) != 1:
                return False

        return True

    @staticmethod
    def __find_perfect_classification(data_set):
        pass

    @staticmethod
    def __get_classified_list(D):
        return [item[-1] for item in D]

    def tree_generate(self):
        root = TreeNode()
        return self.__tree_generate(self.data_set, root, None)


if __name__ == "__main__":
    builder = DecisionTreeBuilder()
    builder.tree_generate()

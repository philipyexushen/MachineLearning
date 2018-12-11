from Common import *
from DataManager import *
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import operator

def check_if_same_attribute(data_set):
    for index in range(len(data_set)):
        data_column = [item[index] for item in data_set]
        if len(set(data_column)) != 1:
            return False

    return True

def get_classified_list(D):
    return [item[-1] for item in D]

def find_perfect_classification(classified_list):
    ret =  Counter(classified_list).most_common(1)
    return ret[0][0]

def caculate_EntD(data_set):
    classified_list = get_classified_list(data_set)

    EntD = 0
    total_size = len(data_set)
    if len(classified_list) == 0:
        return 0

    for count in Counter(classified_list).values():
        EntD = EntD - count / total_size * np.log2(count / total_size)

    return EntD

def slice_data_set(data_set, attribute_index, property_name):
    data_column = [item[attribute_index] for item in data_set]
    index_list = [i for i, item in enumerate(data_column) if item == property_name]
    return [data_set[index] for index in index_list]

class TreeNode:
    def __init__(self, parent):
        self.__category = None
        self.__division_attribute = None
        self.__child = []
        self.__parent = parent

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

    @property
    def parent(self):
        return self.__parent


class DecisionTreeBuilder:
    def __init__(self):
        mgr = ExcelDataManager()
        self.__data_set = mgr.fetch()

        attribute_list = mgr.get_all_title()
        self.__attribute_list = attribute_list[1:len(attribute_list) - 1]

    @property
    def data_set(self):
        return self.__data_set

    @property
    def attribute_list(self):
        return self.__attribute_list

    def __tree_generate(self, data_set, root:TreeNode, attribute_index_list):
        """
        :param data_set: 数据集
        :param attribute_list: 属性集（比如西瓜数据集的色泽，根蒂，敲声，纹理等）
        :param root: 决策树当前根节点
        :return: 决策树当前根节点
        """
        if len(data_set) == 0:
            assert "wrong size"
            return root

        classified_list = get_classified_list(data_set)
        # 看下数据集的元素是不是都是同一个类别的，如果都是同一个类别的，直接返回就好了
        if len(set(classified_list)) == 1:
            root.division_attribute = None
            root.category = classified_list[0]
            return root
        # 如果数据集的各属性的取值都是一样的
        if check_if_same_attribute(data_set):
            root.category =  find_perfect_classification(classified_list)
            return root

        total_EntD = caculate_EntD(data_set)

        GainD_list = []
        # 下面开始找最优划分节点
        for attribute_index in range(len(data_set[0])):
            data_column = [item[attribute_index] for item in data_set]
            properties = list(set(data_column))
            properties_count = Counter(data_column)

            GainD = total_EntD
            for property_name in properties:
                sub_data_set = slice_data_set(data_set, attribute_index, property_name)
                EntD = caculate_EntD(sub_data_set)
                GainD = GainD - len(sub_data_set) / len(data_column) * EntD

            GainD_list.append(GainD)

        # 找到最好的划分节点，开始递归划分
        maxGainD, maxGainD_attribute_index = max(enumerate(GainD_list), key=operator.itemgetter(1))
        data_column = [item[maxGainD_attribute_index] for item in data_set]
        properties = list(set(data_column))

        root.division_attribute = self.attribute_list[maxGainD_attribute_index]
        attribute_index_list_tmp = attribute_index_list.remove(maxGainD_attribute_index)
        for property_name in properties:
            child = TreeNode(root)
            root.child.append(child)

            sub_data_set = slice_data_set(data_set, maxGainD_attribute_index, property_name)
            # 如果分支节点为空，那么直接创造一个子节点，并且标记为D样本中最多的类
            if len(sub_data_set) == 0:
                child.division_attribute = None # 禁止划分，归类
                child.category = find_perfect_classification(get_classified_list(sub_data_set))
            # 否则，递归操作
            else:
                self.__tree_generate(sub_data_set, child, attribute_index_list_tmp)

        return root


    def tree_generate(self):
        root = TreeNode(None)
        attribute_index_list = list(range(self.__attribute_list))
        return self.__tree_generate(self.data_set, root, attribute_index_list)


if __name__ == "__main__":
    builder = DecisionTreeBuilder()
    builder.tree_generate()

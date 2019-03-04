from DecisionTree.Common import *
from DecisionTree.DataManager import *
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import operator
from enum import Enum, auto

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

def calculate_EntD(data_set):
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
        self.__property_name = str()

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
    def property_name(self):
        return self.__property_name

    @property_name.setter
    def property_name(self, value):
        self.__property_name = value

    @property
    def child(self):
        return self.__child

    @property
    def parent(self):
        return self.__parent

    @staticmethod
    def create_node():
        pass

class TreeType(Enum):
    Normal = auto()
    PrePruning = auto()
    PostPruning = auto()

class DecisionTreeBuilder:
    def __init__(self):
        training_data = ExcelDataManager(sheet_name="V1_Training")
        self.__training_data_set = training_data.fetch()
        validation_data = ExcelDataManager(sheet_name="V1_Validation")
        self.__validation_data = validation_data.fetch()

        attribute_list = training_data.get_all_title()
        self.__attribute_list = attribute_list[:len(attribute_list) - 1]
        self.__attribute_list_flag = [False]*len(attribute_list)

    @property
    def training_data_set(self):
        return self.__training_data_set

    @property
    def attribute_list(self):
        return self.__attribute_list

    @staticmethod
    def __pre_generate_tree(data_set, root):
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

        return None

    def __generate_gainD_list(self, data_set):
        total_EntD = calculate_EntD(data_set)
        GainD_list = []
        for attribute_index in range(len(data_set[0]) - 1):
            if self.__attribute_list_flag[attribute_index]:
                continue
            data_column = [item[attribute_index] for item in data_set]
            properties = list(set(data_column))

            GainD = total_EntD
            for property_name in properties:
                sub_data_set = slice_data_set(data_set, attribute_index, property_name)
                EntD = calculate_EntD(sub_data_set)
                GainD = GainD - len(sub_data_set) / len(data_column) * EntD

            GainD_list.append((GainD, attribute_index))

        return GainD_list


    def __tree_generate(self, data_set, root:TreeNode):
        """
        :param data_set: 数据集
        :param root: 决策树当前根节点
        :return: 决策树当前根节点
        """
        ret = self.__pre_generate_tree(data_set, root)
        if ret is not None:
            return ret

        GainD_list = self.__generate_gainD_list(data_set)
        # 找到最好的划分节点，开始递归划分
        # 普通的决策树不需要
        max_index, maxGainD = max(enumerate(GainD_list), key=lambda x : x[1])
        maxGainD_attribute_index = GainD_list[max_index][1]
        data_column = [item[maxGainD_attribute_index] for item in data_set]
        properties = list(set(data_column))

        root.division_attribute = self.attribute_list[maxGainD_attribute_index]
        self.__attribute_list_flag[maxGainD_attribute_index] = True
        for property_name in properties:
            child = TreeNode(root)
            root.child.append(child)
            child.property_name = property_name

            sub_data_set = slice_data_set(data_set, maxGainD_attribute_index, property_name)
            # 如果分支节点为空，那么直接创造一个子节点，并且标记为D样本中最多的类
            if len(sub_data_set) == 0:
                child.division_attribute = None # 禁止划分，归类
                child.category = find_perfect_classification(get_classified_list(sub_data_set))
            # 否则，递归操作
            else:
                self.__tree_generate(sub_data_set, child)

        self.__attribute_list_flag[maxGainD_attribute_index] = False
        return root

    def __pre_pruning_tree_generate(self, data_set, root: TreeNode):
        """
        :param data_set: 数据集 
        :param root: 决策树当前根节点
        :return: 决策树当前根节点
        """
        ret = self.__pre_generate_tree(data_set, root)
        if ret is not None:
            return ret

        GainD_list = self.__generate_gainD_list(data_set)
        max_index, maxGainD = max(enumerate(GainD_list), key=lambda x: x[1])
        GainD_list = sorted(GainD_list, key=lambda x: x[0], reverse=True)

        # 对于预裁剪决策树，要每个最大值都要检测下
        for GainD_list_item in GainD_list:
            if GainD_list_item != maxGainD:
                break

            maxGainD_attribute_index = GainD_list_item[0]
            root.division_attribute = self.attribute_list[maxGainD_attribute_index]
            self.__attribute_list_flag[maxGainD_attribute_index] = True

            data_column = [item[maxGainD_attribute_index] for item in data_set]
            properties = list(set(data_column))

            # 如果不进行划分，那么我们分别考虑其属于什么类别
            for cls in get_classified_list(self.training_data_set):
                # ...算了不想写了，写下去也是个体力活，对于预减枝，因为当前划分虽然并不能提升泛化性能，但是并不能保证后面的也不能提高泛化性能泛化
                pass

            for property_name in properties:
                child = TreeNode(root)
                root.child.append(child)
                child.property_name = property_name

                sub_data_set = slice_data_set(data_set, maxGainD_attribute_index, property_name)
                # 如果分支节点为空，那么直接创造一个子节点，并且标记为D样本中最多的类
                if len(sub_data_set) == 0:
                    child.division_attribute = None  # 禁止划分，归类
                    child.category = find_perfect_classification(get_classified_list(sub_data_set))
                # 否则，递归操作
                else:
                    self.__tree_generate(sub_data_set, child)

            self.__attribute_list_flag[maxGainD_attribute_index] = False


    def tree_generate(self, tree_type = TreeType.Normal):
        root = TreeNode(None)
        if tree_type is TreeType.Normal:
            return self.__tree_generate(self.training_data_set, root)
        elif tree_type is TreeType.PrePruning:
            return self.__pre_pruning_tree_generate(self.training_data_set, root)

if __name__ == "__main__":
    builder = DecisionTreeBuilder()
    root =  builder.tree_generate(TreeType.Normal)
    pass

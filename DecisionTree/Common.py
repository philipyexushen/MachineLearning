import numpy as np

def lower_bound(vector, val, condition):
    left = 0
    right = len(vector)

    while left < right:
        mid = int(left + (right - left) / 2)
        if condition(vector[mid], val):
            left = mid + 1
        else:
            right = mid

    return left

def upper_bound(vector, val, condition):
    left = 0
    right = len(vector)

    while left < right:
        mid = int(left + (right - left) / 2)
        if condition(val,vector[mid]):
            right = mid
        else:
            left = mid + 1

    return left

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

def load_data_set2(file_name):
    num_feat = len(open(file_name).readline().split('\t')) #get number of fields
    data_mat = []

    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr =[]
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)

    return data_mat




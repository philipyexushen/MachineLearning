from DataManager import *
from Common import *

def get_position_and_negative_sum(data_set, left, right):
    sum_pos, sum_neg = 0, 0
    for i in range(left, right):
        if data_set[i][9] == "是":
            sum_pos += 1
        else:
            sum_neg += 1
    return sum_pos, sum_neg


def find_gain_rate(column_index):
    mgr = ExcelDataManager()
    data_set = mgr.fetch()
    data_set = sorted(data_set, key=lambda item: item[column_index])
    print(data_set)

    Ta = [round((float(data_set[i - 1][column_index]) + float(data_set[i][column_index])) / 2, 3) for i in range(1, len(data_set))]

    sum_pos, sum_neg = get_position_and_negative_sum(data_set, 0, len(data_set))
    EntD = -sum_pos / len(data_set) * np.log2(sum_pos / len(data_set)) - sum_neg / len(data_set) * np.log2(
        sum_neg / len(data_set))

    GainD_list = []
    for i, t in enumerate(Ta):
        index = i + 1

        GainD = EntD
        sum_pos, sum_neg = get_position_and_negative_sum(data_set, 0, index)

        if sum_pos != 0 and sum_neg != 0:
            GainD -= index / len(data_set) * (
                        -sum_pos / index * np.log2(sum_pos / index) - sum_neg / index * np.log2(sum_neg / index))

        sum_pos, sum_neg = get_position_and_negative_sum(data_set, index, len(data_set))

        if sum_pos != 0 and sum_neg != 0:
            GainD -= (len(data_set) - index) / len(data_set) * (
                        -sum_pos / (len(data_set) - index) * np.log2(sum_pos / (len(data_set) - index)) \
                        - sum_neg / (len(data_set) - index) * np.log2(sum_neg / (len(data_set) - index)))

        GainD_list.append([GainD, t])

    result = max(GainD_list, key=lambda x: x[0])
    print(result)
    return result


if __name__ == "__main__":
    find_gain_rate(4)










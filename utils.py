import numpy as np
import xlrd


# 将Excel表格中不含第一行表头的所有数据，转化为array形式
def sheet_to_array(sheet, r_begin, r_end, c_begin, c_end):
    r = r_end - r_begin + 1
    c = c_end - c_begin + 1
    arr = np.empty([r, c])
    for i in range(r):
        for j in range(c):
            value = sheet.cell_value(r_begin + i, c_begin + j)
            arr[i, j] = value
    return arr


# 对整个array进行归一化， 用于归一化训练时输入的X
def norm(array, min_values, max_values):
    array = array.copy()  # array是可修改类型变量，需要进行深拷贝
    c = array.shape[1]
    for i in range(c):  # 标签不归一化
        min_v = min_values[i]
        max_v = max_values[i]
        array[:, i] = (array[:, i] - min_v) / (max_v - min_v)
    return array


# 对整个array进行逆归一化，还原数据，用于逆归一化训练和预测时输入的Y
def back_norm(array, min_values, max_values):
    array = array.copy()
    c = array.shape[1]
    for i in range(c):
        min_v = min_values[i]
        max_v = max_values[i]
        array[:, i] = array[:, i] * (max_v - min_v) + min_v
    return array


# 获取Excel表格每一列的最值
def get_values(excel, sheet):
    workbook = xlrd.open_workbook(excel)
    table = workbook.sheet_by_name(sheet)
    data = sheet_to_array(table)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    return min_values, max_values


# 根据输入数据计算效益
def calculate_benefits(square, fish1, fish2, kind):
    N = np.array([0.292, 0.648, 0.968])  # 氮输出价值
    P = np.array([0.323, 0.716, 1.069])  # 处理磷费用
    Algae = np.array([0.497, 1.031, 1.507])  # 藻类消耗价值
    C = np.array([0.008, 0.017, 0.025])  # 碳汇输出价值
    fishes = (fish1 + fish2) * square
    error = 0.10
    benefits_N = np.round([N[kind] * fishes * (1 - error), N[kind] * fishes * (1 + error)], 2)
    benefits_P = np.round([P[kind] * fishes * (1 - error), P[kind] * fishes * (1 + error)], 2)
    benefits_Algae = np.round([Algae[kind] * fishes * (1 - error), Algae[kind] * fishes * (1 + error)])
    benefits_C = np.round([C[kind] * fishes * (1 - error), C[kind] * fishes * (1 + error)])
    # print("预计产生氮输出价值：{}-{}元".format())
    # print("预计处理磷费用：{}-{}元".format())
    # print("预计藻类消耗价值：{}-{}元".format())
    # print("预计碳汇输出价值：{}-{}元".format())
    return benefits_N, benefits_P, benefits_Algae, benefits_C

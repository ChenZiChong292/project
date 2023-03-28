import numpy as np
import xlrd

from predict import predict_fish, predict_index, predict_kind
from utils import sheet_to_array, calculate_benefits

result_dic = {'鲢鱼': 0, '鳙鱼': 0,
              'July': [], 'December': [], 'February': [], 'April': [],
              'N': 0, 'P': 0, 'A': 0, 'C': 0}
root = r'excel\template.xlsx'
workbook = xlrd.open_workbook(root)
fish1 = 'fish1'
fish2 = 'fish2'
# workbook = data['template_excel']  # 获取前端传过来的Excel文件
sheet_name = 'Sheet1'
sheet = workbook.sheet_by_name(sheet_name)
# ----------获取到水库数据，共11个指标 ----------#
# 水库常年平均面积 0
# 总磷 1
# 总氮 2
# 高锰酸钾 3
# 叶绿素a 4
# 浮游植物平均密度 5
# 平均植物量 6
# 浮游动物平均密度 7
# 平均动物量 8
# 浮游植物多样性 9
# 浮游动物多样性 10
# ----------获取到水库数据，共11个指标 ----------#
table = sheet_to_array(sheet, r_begin=1, r_end=sheet.nrows-1, c_begin=1, c_end=sheet.ncols-1)
# ----------预测投鱼量 ----------#
index_6 = np.expand_dims(table[-1, 5:], 0)  # 只需要最后一行的后6个数据
fish1_number = predict_fish(index_6=index_6, fish=fish1)  # 预测鲢鱼
fish2_number = predict_fish(index_6=index_6, fish=fish2)  # 预测鳙鱼
result_dic['鲢鱼'] = fish1_number
result_dic['鳙鱼'] = fish2_number
# ----------预测投鱼量 ----------#

# ----------预测四个月份的指标的指标 ----------#
index_10_July = np.expand_dims(table[1, 1:], 0)
index_10_December = np.expand_dims(table[2, 1:], 0)
index_10_February = np.expand_dims(table[3, 1:], 0)
index_10_April = np.expand_dims(table[4, 1:], 0)
July_index = predict_index(index_10_July, 'July')
December_index = predict_index(index_10_December, 'December')
February_index = predict_index(index_10_February, 'February')
April_index = predict_index(index_10_April, 'April')
result_dic['July'] = July_index.tolist()
result_dic['December'] = December_index.tolist()
result_dic['February'] = February_index.tolist()
result_dic['April'] = April_index.tolist()
# ----------预测四个月份的指标的指标 ----------#

# ----------根据水库营养型预测收益 ----------#
square = table[1, 0]  # 获取水库面积
index_4 = np.expand_dims(table[-1, 1:5], 0)  # 需要最后一行的4个指标
kind = predict_kind(index_4)  # 预测水库营养型
benefits_N, benefits_P, benefits_Algae, benefits_C = calculate_benefits(square, fish1_number, fish2_number, kind)
result_dic['N'] = benefits_N.tolist()
result_dic['P'] = benefits_P.tolist()
result_dic['A'] = benefits_Algae.tolist()
result_dic['C'] = benefits_C.tolist()
print(result_dic)

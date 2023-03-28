import numpy as np
from predict import predict_fish, predict_kind, predict_index
from utils import calculate_benefits

np.set_printoptions(suppress=True)
index_6 = np.array([[355.8806, 2.6908, 10231.6667, 4.9528, 3.5320, 3.4793]])  # 6个指标
index_4 = np.array([[0.09, 1.31, 7.27, 6.17]])  # 4个指标
index_10 = np.concatenate((index_4, index_6), axis=1)
index = ["总磷", "总氮", "高锰酸钾指数", "叶绿素", "浮游植物平均密度", "平均植物量", "浮游动物平均密度", "平均密度量", "浮游植物多样性", "浮游动物多样性"]
fish1 = np.round(predict_fish(index_6, "fish1"), 2)  # 鲢鱼
fish2 = np.round(predict_fish(index_6, "fish2"), 2)  # 鳙鱼
kind = predict_kind(index_4)  # 营养型
square = 1600
benefits_N, benefits_P, benefits_Algae, benefits_C = calculate_benefits(square, fish1, fish2, kind)
print("预计产生氮输出价值：{}-{}元".format(benefits_N[0], benefits_N[1]))
print("预计处理磷费用：{}-{}元".format(benefits_P[0], benefits_P[1]))
print("预计藻类消耗价值：{}-{}元".format(benefits_Algae[0], benefits_Algae[1]))
print("预计碳汇输出价值：{}-{}元".format(benefits_C[0], benefits_C[1]))

month = ['July', 'December', 'February', 'April']
for m in month:
    index = predict_index(index_10, i)
    print("该水库全年指标为{}".format(index_10))
    print("index")



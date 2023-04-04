import matplotlib.pyplot as plt
import numpy as np
import torch
import xlrd
import time
from torch.utils.data import DataLoader
from dataread import MyRegressData
from model import IndexModel
from utils import sheet_to_array, norm


def train(optimizer):
    total_loss_per_epoch = 0
    prediction_list = []
    for data_set in train_data:
        x, y = data_set
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        loss = loss_function(prediction, y)
        prediction_array = prediction.cuda().data.cpu().numpy()
        size = np.size(prediction_array)
        prediction_list.append(prediction_array.reshape(size))
        total_loss_per_epoch = total_loss_per_epoch + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss_per_epoch, prediction_list


if __name__ == '__main__':
    root = "../excel/index.xlsx"
    X_sheet_name = "data_April"
    Y_sheet_name = "label_April"
    workbook = xlrd.open_workbook(root)
    X_table = workbook.sheet_by_name(X_sheet_name)
    Y_table = workbook.sheet_by_name(Y_sheet_name)
    X = sheet_to_array(X_table, 1, X_table.nrows - 1, 0, X_table.ncols - 1)
    Y = sheet_to_array(Y_table, 1, Y_table.nrows - 1, 0, Y_table.ncols - 1)
    X_without_label = X[:, :-1]
    Y_without_label = Y[:, :-1]
    min_values_x = np.min(X_without_label, axis=0)
    max_values_x = np.max(X_without_label, axis=0)
    min_values_y = np.min(Y_without_label, axis=0)
    max_values_y = np.max(Y_without_label, axis=0)
    x = norm(X_without_label, min_values_x, max_values_x)
    y = norm(Y_without_label, min_values_y, max_values_y)
    train_set = MyRegressData(x, y)
    train_data = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    layer = [10, 15, 12, 10]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = IndexModel(layer).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss().to(device)
    epoch = 300
    loss_list = []
    prediction_list = []
    start_time = time.time()
    for i in range(epoch):
        loss_train, prediction = train(optimizer)  # 输出的loss是一个float数，prediction是一个batch*1的二维数组列表
        prediction = np.array(prediction).reshape(np.size(prediction))
        print("第{}轮训练结束".format(i + 1))
        loss_list.append(loss_train)  # 插入每一轮的损失值，10*1
        prediction_list.append(prediction)  # 插入每一轮的预测值，epoch*size（true_Y）
    end_time = time.time()
    print("训练结束，共耗时{}s".format(round(end_time - start_time, 2)))
    # print(loss_list)
    # torch.save(model.state_dict(), "../weight/April.pt")
    plt.figure(1)
    plt.plot(range(epoch), loss_list)
    x_axis = np.array(range(300))
    plt.show()

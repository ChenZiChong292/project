import numpy as np
import torch
import time
import xlrd
from torch import nn
from torch.utils.data import DataLoader
from dataread import MyPredictData
from model import KindModel
from utils import norm, sheet_to_array
import matplotlib.pyplot as plt


root = "../excel/kind.xlsx"
workbook = xlrd.open_workbook(root)

# ---------设置训练集的参数-----------#
train_data_sheet_name = "train_data"
train_table = workbook.sheet_by_name(train_data_sheet_name)
train_data = sheet_to_array(train_table, 1, train_table.nrows - 1, 0, train_table.ncols - 1)
train_data_label = np.expand_dims(train_data[:, -1], axis=1)
train_data_without_label = train_data[:, :-1]
min_values_train_data = np.min(train_data_without_label, axis=0)
max_values_train_data = np.max(train_data_without_label, axis=0)
norm_train_data = norm(train_data_without_label, min_values_train_data, max_values_train_data)
train_set = DataLoader(MyPredictData(norm_train_data, train_data_label), 4, shuffle=True, num_workers=0)
len_data_train = len(norm_train_data)

# ---------设置测试集的参数-----------#
test_data_sheet_name = "test_data"
test_table = workbook.sheet_by_name(test_data_sheet_name)
test_data = sheet_to_array(test_table, 1, test_table.nrows - 1, 0, test_table.ncols - 1)
test_data_label = np.expand_dims(test_data[:, -1], axis=1)
test_data_without_label = test_data[:, :-1]
min_values_test_data = np.min(test_data_without_label, axis=0)
max_values_test_data = np.max(test_data_without_label, axis=0)
norm_test_data = norm(test_data_without_label, min_values_test_data, max_values_test_data)
test_set = DataLoader(MyPredictData(norm_test_data, test_data_label), 2, shuffle=True, num_workers=0)
len_data_test = len(norm_test_data)

# ---------设置网络的参数-----------#
layer = [4, 8, 3]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 可以选择GPU编号
device = torch.device("cpu")
model = KindModel(layer).to(device)
loss_function = nn.CrossEntropyLoss().to(device)
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr)
number_of_train = 0
number_of_test = 0
epoch = 1000


def train(optimizer, len_data_train):
    total_loss_per_epoch = 0
    accuracy = 0
    total_accuracy = 0
    for data_set in train_set:
        datas, targets = data_set
        datas = datas.to(device)
        targets = targets.squeeze()
        targets = targets.to(device)
        outputs = model(datas)
        loss = loss_function(outputs, targets)
        total_loss_per_epoch = loss.item() + total_loss_per_epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = (outputs.argmax(1) == targets).sum().item()
        total_accuracy = total_accuracy + accuracy
    return total_loss_per_epoch, total_accuracy / len_data_train


def test(len_data_test):
    total_loss_per_epoch = 0
    accuracy = 0
    total_accuracy = 0
    with torch.no_grad():
        for data_set in test_set:
            datas, targets = data_set
            datas = datas.to(device)
            targets = targets.squeeze()
            targets = targets.to(device)
            outputs = model(datas)
            loss = loss_function(outputs, targets)
            total_loss_per_epoch = loss.item() + total_loss_per_epoch
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy = total_accuracy + accuracy
    return total_loss_per_epoch, total_accuracy / len_data_test


if __name__ == '__main__':
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    gap = 50
    start_time = time.time()
    print("训练开始")
    for i in range(epoch):
        if (i + 1) % gap == 0:
            print("第{}轮训练开始".format(i + 1))
        loss_train, acc_train = train(optimizer, len_data_train)
        loss_test, acc_test = test(len_data_test)
        if (i + 1) % gap == 0:
            print("第{}轮训练集上的损失为{}".format(i + 1, loss_train))
            print("第{}轮测试集上的损失为{}".format(i + 1, loss_test))
            print("第{}轮训练集上的正确率为{}".format(i + 1, acc_train))
            print("第{}轮测试集上的正确率为{}".format(i + 1, acc_test))
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_loss.append(loss_test)
            test_acc.append(acc_test)
    end_time = time.time()
    # torch.save(model.state_dict(), "../weight/kind.pt")
    print("耗时{}s".format(round(end_time - start_time, 2)))
    plt.figure(1)
    # plt.plot(range(1, epoch + 1, gap), train_loss, label='train set')
    # plt.plot(range(1, epoch + 1, gap), test_loss, label='test set')
    # plt.xlabel("The number of training and testing")
    # plt.ylabel("Loss")
    # plt.figure(2)
    # plt.plot(range(1, epoch + 1, gap), train_acc, label='train_acc')
    # plt.plot(range(1, epoch + 1, gap), test_acc, label='test_acc')
    plt.plot(range(1, epoch + 1, gap), train_acc)
    plt.xlabel("The number of training and testing")
    plt.ylabel("Accuracy")
    # plt.legend()
    plt.show()

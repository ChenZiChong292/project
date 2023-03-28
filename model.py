from torch import nn


# KindModel用于预测营养型
class KindModel(nn.Module):
    def __init__(self, layer):
        super(KindModel, self).__init__()
        self.fc1 = nn.Linear(layer[0], layer[1], True)
        self.fc2 = nn.Linear(layer[1], layer[2], True)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.soft(x)
        # x = x.argmax(1)
        return x


# FishModel用于预测投鱼量
class FishModel(nn.Module):
    def __init__(self, layer):
        super(FishModel, self).__init__()
        self.fc1 = nn.Linear(layer[0], layer[1], True)
        self.fc2 = nn.Linear(layer[1], layer[2], True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# IndexModel用于预测指标的变化
class IndexModel(nn.Module):
    def __init__(self, layer):
        super(IndexModel, self).__init__()
        self.fc1 = nn.Linear(layer[0], layer[1], True)
        self.fc2 = nn.Linear(layer[1], layer[2], True)
        self.fc3 = nn.Linear(layer[2], layer[3], False)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tan(x)
        x = self.fc3(x)
        return x


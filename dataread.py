import torch
from torch.utils.data import Dataset


class MyPredictData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx, :]).type(torch.float32)
        label = torch.from_numpy(self.label[idx, :]).type(torch.LongTensor)
        return data, label

    def __len__(self):
        return len(self.data)


class MyRegressData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx, :]).type(torch.float32)
        y = torch.from_numpy(self.Y[idx, :]).type(torch.float32)
        return x, y

    def __len__(self):
        return len(self.X)

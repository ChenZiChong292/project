import numpy as np
import torch
from model import FishModel, IndexModel, KindModel
from utils import norm, back_norm


# 预测营养型
def predict_kind(index_4):
    min_values = [0.03, 0.57, 1.01, 4.95]
    max_values = [0.12, 2.64, 7.63, 7.74]
    norm_x = norm(index_4, min_values, max_values)
    x = torch.from_numpy(norm_x).type(torch.float32)
    device = torch.device("cpu")
    layer = [4, 8, 3]
    model = KindModel(layer).to(device)
    weight = torch.load('weight/kind.pt', map_location=device)
    model.load_state_dict(weight)
    y = model(x).argmax(1).item()
    return y


# 预测鱼
def predict_fish(index_6, fish):
    min_values_x = [50.34, 0.47, 676.16, 0.8, 3.28, 2.33]
    max_values_x = [396.4, 2.87, 10522.8, 5.19, 4.21, 3.65]
    device = torch.device("cpu")
    layer = [6, 6, 1]
    model = FishModel(layer).to(device)
    if fish == 'fish1':
        min_values_y = [2.11]
        max_values_y = [8.27]
        norm_X = norm(index_6, min_values_x, max_values_x)
        x = torch.from_numpy(norm_X).type(torch.float32)
        weight = torch.load('weight/fish1.pt', map_location=device)
    else:
        min_values_y = [12.03]
        max_values_y = [46.82]
        norm_X = norm(index_6, min_values_x, max_values_x)
        x = torch.from_numpy(norm_X).type(torch.float32)
        weight = torch.load('weight/fish2.pt', map_location=device)
    model.load_state_dict(weight)
    out = np.array([model(x).item()])
    out = out.reshape(1, 1)
    y = back_norm(out, min_values_y, max_values_y)[0][0]
    return y


# 预测指标
def predict_index(index_10, month):
    device = torch.device("cpu")
    if month == 'July':
        min_values_x = [0.039, 0.577, 1.298, 4.125, 59.471, 0.706, 1473.394, 0.96, 2.598, 2.572]
        max_values_x = [0.178, 2.234, 9.267, 13.883, 267.572, 3.622, 22310.917, 6.702, 4.564, 3.887]
        min_values_y = [0.036, 0.486, 1.142, 4.031, 61.8, 0.562, 1162.017, 1.011, 2.532, 2.469]
        max_values_y = [0.153, 2.263, 8.856, 14.19, 197.585, 2.674, 17587.833, 5.344, 4.252, 3.717]
        weight = torch.load('weight/July.pt', map_location=device)
    elif month == 'December':
        min_values_x = [0.009, 0.444, 0.69, 2.705, 36.308, 0.437, 182.05, 0.349, 3.138, 1.917]
        max_values_x = [0.126, 2.776, 7.928, 5.908, 419.728, 2.581, 11000.521, 5.86, 4.569, 3.364]
        min_values_y = [0.009, 0.473, 0.695, 2.231, 31.427, 0.304, 99.657, 0.247, 3.188, 1.86]
        max_values_y = [0.113, 3.081, 8.783, 8.558, 152.275, 1.414, 3760.669, 4.847, 4.561, 3.504]
        weight = torch.load('weight/December.pt', map_location=device)
    elif month == 'February':
        min_values_x = [0.009, 0.516, 0.695, 6.3, 44.41, 0.315, 468.232, 0.112, 3.171, 1.963]
        max_values_x = [0.21, 2.821, 8.474, 9.106, 267.144, 3.84, 11020.712, 3.276, 4.413, 3.67]
        min_values_y = [0.009, 0.49, 0.752, 5.768, 24.805, 0.208, 417.366, 0.094, 3.262, 2.071]
        max_values_y = [0.18, 2.626, 5.504, 8.73, 151.528, 2.542, 8912.523, 3.076, 4.37, 3.6]
        weight = torch.load('weight/February.pt', map_location=device)
    else:
        min_values_x = [0.018, 0.765, 0.785, 4.995, 43.218, 0.255, 341.343, 0.235, 2.607, 1.607]
        max_values_x = [0.094, 2.694, 7.549, 7.695, 840.248, 8.236, 14376.908, 6.555, 4.29, 3.812]
        min_values_y = [0.012, 0.751, 0.758, 5.147, 28.842, 0.178, 224.979, 0.137, 2.394, 1.787]
        max_values_y = [0.082, 2.656, 7.432, 8.239, 744.923, 6.101, 11892.171, 6.101, 3.925, 3.754]
        weight = torch.load('weight/April.pt', map_location=device)
    layer = [10, 15, 12, 10]
    model = IndexModel(layer).to(device)
    norm_X = norm(index_10, min_values_x, max_values_x)
    x = torch.from_numpy(norm_X).type(torch.float32)
    model.load_state_dict(weight)
    out = np.array(model(x).detach())
    y = back_norm(out, min_values_y, max_values_y).reshape(out.size)
    return y

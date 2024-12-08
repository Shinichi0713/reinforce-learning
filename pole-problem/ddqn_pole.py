
import gym
import numpy as np
import time, os
import torch
import torch.nn as nn
from collections import deque


# フーバーロス
class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, y_true, y_pred):
        err = y_true - y_pred
        cond = torch.abs(err) < 1.0
        L2 = 0.5 * err ** 2
        L1 = (torch.abs(err) - 0.5)
        loss = torch.where(cond, L2, L1)
        return torch.mean(loss)



if __name__ == "__main__":
    x_true = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    x_pred = torch.tensor([0.5, 2.5, 2.5, 4], dtype=torch.float32)

    criterion = HuberLoss()
    loss = criterion(x_true, x_pred)
    print(loss)

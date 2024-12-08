
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

# 経験再生用のメモリ
class Memory:
    def __init__(self, size_max=1000):
        self.buffer = deque(maxlen=size_max)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
    
    def len(self):
        return len(self.buffer)

# エージェント
class Agent():
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=100):
        # モデルはkeras製
        self.model=nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        

if __name__ == "__main__":
    x_true = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    x_pred = torch.tensor([0.5, 2.5, 2.5, 4], dtype=torch.float32)

    criterion = HuberLoss()
    loss = criterion(x_true, x_pred)
    print(loss)

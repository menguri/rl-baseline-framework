import torch
import torch.nn as nn
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=400, hidden2=300, init_w=3e-3, bn_use=True):
        super().__init__()
        self.bn_use = bn_use
        self.bn_in = nn.BatchNorm1d(state_dim)
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        if self.bn_use:  
            x = self.bn_in(x)
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.tanh(self.fc3(x))
        else:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.tanh(self.fc3(x))
        return x
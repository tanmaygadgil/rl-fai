import torch
from torch.autograd import Variable
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torchvision.transforms as T
import numpy as np
from torch.distributions import Categorical
import time

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Network(torch.nn.Module):

    def __init__(self, input_shape, action_space, is_actor = False) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.action_space = action_space
        self.is_actor = is_actor
        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        if is_actor:
            self.fc3 = nn.Linear(FC2_DIMS, self.action_space)
        else:
            self.fc3 = nn.Linear(FC2_DIMS, 1)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.is_actor:
            distribution = Categorical(F.softmax(x, dim=-1))
            return distribution
        else:
            return x
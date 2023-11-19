
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


import sys
sys.path.append('../')


import pdb


class BackboneRL(nn.Module):
    def __init__(self, input_channels, output_dim):   # [txbf, rxbf]
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2,padding=1) # (64, 64, 3) -> (32, 32, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2,padding=1) # (32, 32, 32) -> (16, 16 ,32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # (16,16 ,32 ) -> (8, 8, 32)
        self.fc = nn.Linear(8*8*32, output_dim)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print('shape:', x.shape)
        x = x.view(-1, 8*8*32)
        out = self.fc(x)

        return out






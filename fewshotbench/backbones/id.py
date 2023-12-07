import torch
from torch import nn as nn

class Id(nn.Module):

    def __init__(self, x_dim):
        super().__init__()
        self.final_feat_dim = x_dim

    def forward(self, x):
        return x
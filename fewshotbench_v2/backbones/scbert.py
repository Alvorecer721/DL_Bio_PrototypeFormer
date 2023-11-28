import torch
from torch import nn
from backbones.performer_pytorch import Performer


class scBERT(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8, dim_head=32):
        super().__init__()
        self.performer = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,  
        )
       
    def forward(self, x):
        return self.performer(x)
import torch
from torch import nn as nn

class PEM(nn.Module):

    def __init__(self, x_dim, layer_dim=[64, 64], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = x_dim
        for dim in layer_dim:
            layers.append(
                nn.TransformerEncoderLayer(d_model=in_dim, nhead=2, dim_feedforward=dim, dropout=dropout, activation='relu', norm_first=True)
            )
            in_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.final_feat_dim = layer_dim[-1]

    def forward(self, x):
        x = self.encoder(x)
        return x

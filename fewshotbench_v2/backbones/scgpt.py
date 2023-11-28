import torch
from torch import nn

from scgpt_transformer.model import TransformerModel

class SCGPTBackbone(nn.Module):
    def __init__(self, x_dim, d_model, nhead, d_hid, nlayers, **kwargs):
        super(SCGPTBackbone, self).__init__()
        self.transformer = TransformerModel(x_dim, d_model, nhead, d_hid, nlayers, **kwargs)

    def forward(self, x):
        return self.transformer(values=x)

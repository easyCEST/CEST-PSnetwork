from torch import nn
import torch
import numpy as np


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.sum((inputs - targets) ** 2) / targets.numel()

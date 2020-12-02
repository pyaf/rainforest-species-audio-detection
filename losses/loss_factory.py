from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(**_):
    return torch.nn.CrossEntropyLoss()


def bce(**_):
    return torch.nn.BCEWithLogitsLoss()


def mse_loss(**_):
    return torch.nn.MSELoss()


def l1_loss(**_):
    return torch.nn.L1Loss()


def smooth_l1_loss(**_):
    return torch.nn.SmoothL1Loss()


def focal_loss(**kwargs):
    return FocalLoss(gamma=kwargs['gamma'])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)

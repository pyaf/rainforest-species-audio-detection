import pdb
import copy
import torch
from torch import nn
import pretrainedmodels
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        # pdb.set_trace()
        return torch.cat([self.mp(x), self.ap(x)], 1)


def resnext101_32x16d(out_features):
    """[1]"""
    model = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x16d_wsl")
    for params in model.parameters():
        params.requires_grad = False

    model.fc = nn.Linear(in_features=2048, out_features=out_features, bias=True)
    # every new layer added, has requires_grad = True

    return model


def efficientNet(name, out_features, pretrained="imagenet"):
    """name like: `efficientnet-b5`
    [2]
    """
    if pretrained:
        model = EfficientNet.from_pretrained(name)
    else:
        model = EfficientNet.from_name(name)

    for params in model.parameters():
        params.requires_grad = False

    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    return model


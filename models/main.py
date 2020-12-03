import pdb
import types
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from resnest.torch import resnest50


from .pretrained import *
from .resnet import *


def get_model(model_name, num_classes=1, pretrained="imagenet"):


    if model_name == "resnext101_32x16d":
        return resnext101_32x16d(num_classes)

    elif model_name.startswith("efficientnet"):
        return efficientNet(model_name, num_classes, pretrained)

    elif model_name == 'resnest50':
        return resnest(num_classes, pretrained=pretrained)

    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    for params in model.parameters():
        params.requires_grad = False

    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=True
    )
    return model


def resnest(num_classes, pretrained=True):
    model = resnest50(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes)
    )

    return model


def get_pretrainedmodels(model_name='resnet18', num_outputs=None, pretrained=True, **_):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)

    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs,
                                      kernel_size=1, bias=True)
    else:
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_outputs)

    return model





""" footnotes

[1]: model.avgpool is already AdapativeAvgPool2d, and model's forward method handles flatten and stuff. So here I'm just adding a trainable the last fc layer, after few epochs the model's all layers will be set required_grad=True
Apart from that this model is trained on instagram images, remove imagenet mean and std, only gotta divide by 255, so mean=0,std=1

[2]: efficientnet models are trained on imagenet, so make sure mean and std are of imagenet.
"""

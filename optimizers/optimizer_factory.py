from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim as optim
from .radam import RAdam


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    #print('weight decay:', weight_decay)
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                      amsgrad=amsgrad)


def SGD(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, lr=float(lr), momentum=momentum, weight_decay=float(weight_decay),
                     nesterov=nesterov)


def get_optimizer(config, parameters):

    #import pdb; pdb.set_trace()
    f = globals().get(config.optimizer.name)
    return f(parameters, **config.optimizer.params)

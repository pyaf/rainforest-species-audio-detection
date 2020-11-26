import os
import pdb
import cv2
import time
import json
import torch
import random
import scipy
import logging
import traceback
import numpy as np
from datetime import datetime

# from config import HOME
from tensorboard_logger import log_value, log_images
from torchnet.meter import ConfusionMeter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from pycm import ConfusionMatrix
from .extras import *


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.predictions = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.thresholds = 0.5
        self.save_folder = os.path.join(save_folder, "logs")

    def update(self, targets, outputs):
        """targets, outputs are detached CUDA tensors"""
        targets = targets.type(torch.LongTensor).flatten()
        outputs = (outputs > self.thresholds).type(torch.LongTensor).flatten()

        #pdb.set_trace()
        self.targets.extend(targets.tolist())
        self.predictions.extend(outputs.tolist())

    def get_cm(self):
        targets = np.array(self.targets)
        predictions = np.array(self.predictions)
        cm = ConfusionMatrix(targets, predictions)
        return cm


def epoch_log(opt, log, tb, phase, epoch, epoch_loss, meter, start):
    cm = meter.get_cm()

    lr = opt.param_groups[-1]["lr"]
    # take care of base metrics
    acc, tpr, ppv, f1, cls_tpr, cls_ppv, cls_f1 = get_stats(cm)
    log(
        "ACC: %0.4f | TPR: %0.4f | PPV: %0.4f | F1: %0.4f"
        % (acc, tpr, ppv, f1)
    )
    log(f"Class TPR: {cls_tpr}")
    log(f"Class PPV: {cls_ppv}")
    log(f"Class F1: {cls_f1}")
    cm.print_normalized_matrix()
    log(f"lr: {lr}")

    # tensorboard
    logger = tb[phase]
    for cls in cls_tpr.keys():
        logger.log_value("TPR_%s" % cls, float(cls_tpr[cls]), epoch)
        logger.log_value("PPV_%s" % cls, float(cls_ppv[cls]), epoch)
        logger.log_value("F1_%s" % cls, float(cls_f1[cls]), epoch)

    if phase == "train":
        logger.log_value("lr", lr, epoch)

    logger.log_value("loss", epoch_loss, epoch)
    logger.log_value(f"ACC", acc, epoch)
    logger.log_value(f"TPR", tpr, epoch)
    logger.log_value(f"PPV", ppv, epoch)


    # save pycm confusion
    obj_path = os.path.join(meter.save_folder, f"cm{phase}_{epoch}")
    cm.save_obj(obj_path, save_stat=True, save_vector=True)

    return acc


def get_stats(cm):
    acc = cm.overall_stat["Overall ACC"]
    tpr = cm.overall_stat["TPR Macro"]  # [7]
    ppv = cm.overall_stat["PPV Macro"]
    f1 = cm.overall_stat["F1 Macro"]
    cls_tpr = cm.class_stat["TPR"]
    cls_ppv = cm.class_stat["PPV"]
    cls_f1 = cm.class_stat["F1"]

    if tpr is "None":
        tpr = 0  # [8]
    if ppv is "None":
        ppv = 0
    if f1 is "None":
        f1 = 0

    cls_tpr = sanitize(cls_tpr)
    cls_ppv = sanitize(cls_ppv)
    cls_f1 = sanitize(cls_f1)

    return acc, tpr, ppv, f1, cls_tpr, cls_ppv, cls_f1


def sanitize(cls_dict):
    for x, y in cls_dict.items():
        try:
            cls_dict[x] = float("%0.4f" % y)
        except Exception as e:  # [8]
            cls_dict[x] = 0.0
    return cls_dict


def check_sanctity(dataloaders):
    phases = dataloaders.keys()
    if len(phases) > 1:
        tnames = dataloaders["train"].dataset.fnames
        vnames = dataloaders["val"].dataset.fnames
        common = [x for x in tnames if x in vnames]
        if len(common):
            print("TRAIN AND VAL SET NOT DISJOINT")
            exit()
    else:
        print("No sanctity check")

"""Footnotes:

[1]: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

[2]: Used in cross-entropy loss, one-hot to single label

[3]: # argmax returns earliest/first index of the maximum value along the given axis
 get_preds ka ye hai ki agar kisi output me zero nahi / sare one hain to 5 nahi to jis index par pehli baar zero aya wahi lena hai, example:
[[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
-> [4, 1, 0, 0]
baki clip karna hai (0, 4) me, we can get -1 for cases with all zeros.

[4]: get_best_threshold is used in the validation phase, during each phase (train/val) outputs and targets are accumulated. At the end of train phase a threshold of 0.5 is used for
generating the final predictions and henceforth for the computation of different metrics.
Now for the validation phase, best_threshold function is used to compute the optimum threshold so that the qwk is minimum and that threshold is used to compute the metrics.

It can be argued ki why are we using 0.5 for train, then, well we used 0.5 for both train/val so far, so if we are computing this val set best threshold, then not only it can be used to best evaluate the model on val set, it can also be used during the test time prediction as it is being saved with each ckpt.pth

[5]: np.array because it's a list and gets converted to np.array in get_best_threshold function only which is called in val phase and not training phase

[6]: It's important to keep these two in np array, else ConfusionMatrix takes targets as strings. -_-

[7]: macro mean average of all the classes. Micro is batch average or sth.

[8]: sometimes initial values may come as "None" (str)

[9]: I'm using base th for train phase, so base_qwk and best_qwk are same for train phase, helps in comparing the base_qwk and best_qwk of val phase with the train one, didn't find a way to plot base_qwk of train with best and base of val on a single plot.
"""

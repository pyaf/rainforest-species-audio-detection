#!/usr/bin/env python
# coding: utf-8
import pdb
from glob import glob
import os
from tensorboard_logger import *
from pycm import *
from argparse import ArgumentParser


def fl(value):
    """fixed digit float value"""
    return "%.4f" % value


def fd(train_dict, val_dict):
    """ return a dict with values in train/val format """
    metric_dict = {}
    for key in train_dict.keys():
        key2 = type(list(val_dict.keys())[0])(key)  # [4]
        metric_dict[key] = f"{fl(train_dict[key])}/{fl(val_dict[key2])}"
    return metric_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--ckpt_path",
        dest="ckpt_path",
        help="Path to the ckpt file",
        metavar="FOLDER",
    )

    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    model_folder = os.path.dirname(ckpt_path)
    ckpt = os.path.basename(ckpt_path)  # ckpt10.pth
    epoch = ckpt.split(".")[0][4:]

    print(f"ckpt_path: {ckpt_path}")
    print(f"ckpt: {ckpt}")
    print(f"epoch: {epoch}")

    train_obj = os.path.join(model_folder, f"logs/base_cmtrain_{epoch}.obj")
    val_obj = os.path.join(model_folder, f"logs/base_cmval_{epoch}.obj")

    train_cm = ConfusionMatrix(file=open(train_obj))
    val_cm = ConfusionMatrix(file=open(val_obj))

    train_dict = json.load(open(train_obj))
    val_dict = json.load(open(val_obj))

    overall_metrics = ["TPR Macro", "PPV Macro", "F1 Macro", "Overall ACC"]
    class_metrics = ["TPR", "PPV", "F1", "AUC"]

    print("\nOverall metrics")
    for metric in overall_metrics:
        print(
            f"{metric}: {fl(train_cm.overall_stat[metric])}/{fl(val_cm.overall_stat[metric])}"
        )

    print("\nClass metrics")
    for metric in class_metrics:
        metric_dict = fd(train_cm.class_stat[metric], val_cm.class_stat[metric])
        print(f"{metric}: {metric_dict}")

    # print a row so that copy pasting to google sheet is easy
    order = []
    row = []
    oc = "Overall ACC"
    order.extend(["loss", "QWK", "ACC"])

    qwk = f"{fl(train_dict['qwk'])}/{fl(val_dict['qwk'])}"
    # base_qwk = f"{fl(val_dict['qwk'])}"
    # qwk = best_qwk + " - " + base_qwk
    loss = f"{fl(train_dict['loss'])}/{fl(val_dict['loss'])}"
    acc = f"{fl(train_cm.overall_stat[oc])}/{fl(val_cm.overall_stat[oc])}"
    row.extend([loss, qwk, acc])

    # metrics which are only overall, not class wise

    # those which are in format of overall train/ overall val <space> class wise train/val
    for om, cm in zip(overall_metrics[:3], class_metrics[:3]):
        order.append(f"{om} {cm}")
        overall = (
            f"{fl(train_cm.overall_stat[om])}/{fl(val_cm.overall_stat[om])} "
        )  # [5]
        classwise = f"{fd(train_cm.class_stat[cm], val_cm.class_stat[cm])}"
        row.append(overall + classwise)

    # only class wise metrics, metric_dict corresponds to last one in class_metrics.
    order.append(class_metrics[-1])
    row.append(str(metric_dict))
    row = ";".join(row)
    print(order)
    print("\n" + row + "\n")

    # just copy paste the printed row, and choose seperator as semi-colon


""" Footnotes

[1]: If input targets to ConfusionMatrix is not numpy it takes those as strings. So, earlier code didn't do that and to tackle those cases we have try except
[2]: Macros is average of all classes. Micro is sth else ;D
[3]: classwise accuracy doesn't average up to Overall ACC, dunno why.
[4]: Due to a bug, many previously trained models were saved with str and int class labels for train val cm obj.
[5]: space is important, for readability
"""

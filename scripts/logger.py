#!/usr/bin/env python
# coding: utf-8


import pdb
from glob import glob
import os
from tensorboard_logger import *
from pycm import *
from argparse import ArgumentParser


def log_all(name, data_list, logger):
    for idx, val in enumerate(data_list):
        logger.log_value(name, val, idx)


def log_overall(folder_name, phase, log_folder, metrics):
    """
    For logging overall stats of train and val cms in seperate folders, so that we can see them on same chart
    folder_name: path of the folder where model weights/logs were stored,
    phase: train/val
    save_folder: place where to log tb files
    metrics: (list) metrics to log
    """
    cms = {}
    cmfiles = glob(os.path.join(folder_name, "logs/*%s*.obj" % phase))
    for i in range(len(cmfiles)):
        cms[f"cm{i}"] = ConfusionMatrix(
            file=open(os.path.join(folder_name, f"logs/cm{phase}_{i}.obj"), "r")
        )
    logger = Logger(os.path.join(log_folder, phase))
    overall_stats = {x: [] for x in metrics}
    for i in range(len(cmfiles)):
        for metric in metrics:
            overall_stats[metric].append(cms["cm%d" % i].overall_stat[metric])
    for metric in metrics:
        metric_name = "overall " + metric if metric != "Overall ACC" else metric
        log_all(metric_name, overall_stats[metric], logger)


def log_class_stats(folder_name, phase, log_folder, metrics):
    """
    Stats of each class plotted under each phase
    folder_name: path of the folder where model weights/logs were stored,
    phase: train/val
    save_folder: place where to log tb files
    metrics: (list) metrics to log
    """
    # pdb.set_trace()
    cms = {}
    cmfiles = glob(os.path.join(folder_name, "logs/*%s*.obj" % phase))
    for i in range(len(cmfiles)):
        cms[f"cm{i}"] = ConfusionMatrix(
            file=open(os.path.join(folder_name, f"logs/cm{phase}_{i}.obj"), "r")
        )
    for class_name in range(5):
        logger = Logger(os.path.join(log_folder, phase, str(class_name)))
        class_stats = {x: [] for x in metrics}
        for i in range(len(cmfiles)):
            for metric in metrics:
                try:
                    value = cms["cm%d" % i].class_stat[metric][class_name]
                except Exception as e:
                    value = cms["cm%d" % i].class_stat[metric][str(class_name)]  # [1]
                class_stats[metric].append(value if value is not "None" else 0)
        for metric in metrics:
            metric_name = "class " + metric
            log_all(metric_name, class_stats[metric], logger)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--folder_name",
        dest="folder_name",
        help="Folder where model logs are stored, example 'weights/6Jul_resnext101_32/'",
        metavar="FOLDER",
    )

    args = parser.parse_args()
    folder_name = args.folder_name
    log_folder = os.path.join(folder_name, "logs")

    # empty previously written tb files by this file

    files = glob(os.path.join(log_folder, "*/*/*"))  # logs/train/1/*
    if len(files):
        print("deleting files:")
        print(files)
    for f in files:
        os.remove(f)

    """log overall statistics"""

    #    overall_metrics = [
    #        "Overall ACC",
    #        "Kappa",
    #        "TPR Micro",
    #        "PPV Micro",
    #        "F1 Micro",
    #        "Cross Entropy",
    #    ]

    overall_metrics = ["TPR Micro", "PPV Micro"]

    print(f"Logging overall metrics: {overall_metrics}")
    log_overall(folder_name, "train", log_folder, overall_metrics)
    log_overall(folder_name, "val", log_folder, overall_metrics)

    """log class statistics"""

    # class_metrics = ["TPR", "TNR", "PPV", "NPV", "FNR", "FPR", "ACC", "F1", "AUC"]
    class_metrics = ["TPR", "PPV"]
    print()
    print(f"Logging class metrics: {class_metrics}")
    log_class_stats(folder_name, "train", log_folder, class_metrics)
    log_class_stats(folder_name, "val", log_folder, class_metrics)
    print("Done!")

    # #### NOTES:
    # * If curve goes to zero, suddenly, they that's a 0 replaced in place of None value


""" Footnotes

[1]: If input targets to ConfusionMatrix is not numpy it takes those as strings. So, earlier code didn't do that and to tackle those cases we have try except
"""

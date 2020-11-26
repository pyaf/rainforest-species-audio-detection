import os
import pdb
import time
import yaml
import random
import pprint
import torch
import numpy as np
import logging
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from matplotlib import pyplot as plt
from tensorboard_logger import log_value, log_images
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from easydict import EasyDict as edict

plt.switch_backend("agg")


def get_parser():
    """Get parser object."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="filepath",
        help="experiment config file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume",
        help="Use when to resume from ckpt.pth",
        action="store_true",
    )  # use -r when to resume, else don't

    args = parser.parse_args()
    return args


def load_cfg(args):
    filepath = args.filepath
    with open(filepath, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    cfg = prepare_config(cfg)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    return cfg

def prepare_config(cfg):
    cfg = edict(cfg)
    cfg.batch_size = edict(cfg.batch_size)
    cfg.home = Path(cfg.home)
    cfg.data_home = Path(cfg.data_home)
    cfg.df_path = Path(cfg.df_path)
    cfg.sample_submission = Path(cfg.sample_submission)
    cfg.data_folder = Path(cfg.data_folder)
    cfg = change_types(cfg)
    return cfg


def change_types(cfg):
    cfg.top_lr = eval(cfg.top_lr)
    cfg.optimizer.params.lr = eval(cfg.optimizer.params.lr)
    return cfg



def save_cfg(cfg, trainer):
    augmentations = trainer.dataloaders["train"].dataset.transform.transforms
    text = (
        f"model_name: {trainer.model_name}\n"
        + f"augmentations: {augmentations}\n"
        + f"criterion: {trainer.criterion}\n"
        + f"optimizer: {trainer.optimizer}\n"
    )
    print(text)
    filepath = trainer.args.filepath
    filename = os.path.basename(filepath)
    cp_file = os.path.join(trainer.save_folder, filename)
    copyfile(filepath, cp_file)


def logger_init(save_folder):
    mkdir(save_folder)
    logging.basicConfig(
        filename=os.path.join(save_folder, "log.txt"),
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    return logger


def plot_ROC(roc, targets, predictions, phase, epoch, folder):
    roc_plot_folder = os.path.join(folder, "ROC_plots")
    mkdir(os.path.join(roc_plot_folder))
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_plot_name = "ROC_%s_%s_%0.4f" % (phase, epoch, roc)
    roc_plot_path = os.path.join(roc_plot_folder, roc_plot_name + ".jpg")
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.legend(["diagonal-line", roc_plot_name])
    fig.savefig(roc_plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # see footnote [1]

    plot = cv2.imread(roc_plot_path)
    log_images(roc_plot_name, [plot], epoch)


def print_time(log, start, string):
    diff = time.time() - start
    log(string + ": %02d:%02d" % (diff // 60, diff % 60))


def adjust_lr(lr, optimizer):
    """ Update the lr of base model
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    print('LR adjusting to %f' % lr)
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = lr
    return optimizer


def iter_log(log, phase, epoch, iteration, epoch_size, loss, start):
    diff = time.time() - start
    log(
        "%s epoch: %d (%d/%d) loss: %.4f || %02d:%02d",
        phase,
        epoch,
        iteration,
        epoch_size,
        loss.item(),
        diff // 60,
        diff % 60,
    )


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    # if not os.path.exists(folder):
    # os.mkdir(folder)


def seed_pytorch(seed=69):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True # slows down the training


def tt(cuda):
    tensor_type = "torch%s.FloatTensor" % (".cuda" if cuda else "")
    torch.set_default_tensor_type(self.tensor_type)


def commit(model_name):
    import subprocess

    cmd1 = "git add ."
    cmd2 = f'git commit -m "{model_name}"'
    process = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(error)
    process = subprocess.Popen(cmd2.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(error)

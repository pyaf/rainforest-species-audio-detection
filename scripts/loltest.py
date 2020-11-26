import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from PIL import Image
import jpeg4py as jpeg
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
from models import get_model
from dataloader import *
from extras import *
from augmentations import *
from utils import *
from image_utils import *
from preprocessing import *


def test_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="filepath",
        help="experiment config file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--epoch_range",
        nargs="+",
        type=int,
        dest="epoch_range",
        help="Epoch to start from",
    )  # usage: -e 10 20
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="test",
    )
    parser.add_argument(
        "-s", "--size", dest="size", help="image size to use", default=256
    )

    return parser


def get_predictions(model, testset, tta):
    """return all predictions on testset in a list"""
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        _, images, _ = batch
        preds = torch.sigmoid(model(images.to(device)))
        preds = preds.detach().tolist()  # [1]
        predictions.extend(preds)
    return np.array(predictions)


if __name__ == "__main__":
    """
    Generates predictions on train/test set using the ckpts saved in the model folder path and saves them in npy_folder in npy format which can be analyses later for different thresholds
    """
    parser = test_parser()
    args = parser.parse_args()
    predict_on = args.predict_on
    eps_range = args.epoch_range
    if len(eps_range) == 1:
        eps_range *= 2  # [1] -> [1, 1]
    start_epoch, end_epoch = eps_range
    cfg = load_cfg(args)

    cfg['phase'] = args.predict_on
    if predict_on == "train":
        cfg['sample_submission'] = cfg['new_df_path']
    if predict_on == "mes":
        cfg['sample_submission'] = cfg['mes_df']
    if predict_on == "idrid":
        cfg['sample_submission'] = cfg['idrid_df']

    tta = 4  # number of augs in tta

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    size = cfg['size']
    test_dataloader = testprovider(cfg)

    model = get_model(cfg['model_name'], cfg['num_classes'], pretrained=None)
    model.to(device)
    model.eval()
    folder = os.path.splitext(os.path.basename(args.filepath))[0]
    model_folder_path = os.path.join( 'weights', folder)
    npy_folder = os.path.join(model_folder_path, f"{predict_on}_npy/{size}")

    mkdir(npy_folder)

    print(f"Saving predictions at: {npy_folder}")
    print(f"From epoch {start_epoch} to {end_epoch}")
    print(f"Using tta: {tta}\n")

    base_th = 0.5
    y_test = pd.read_csv('weights/submission829.csv')['diagnosis'].values
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"Using ckpt{epoch}.pth")
        ckpt_path = os.path.join(model_folder_path, "ckpt%d.pth" % epoch)
        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        preds = get_predictions(model, test_dataloader, tta)
        outputs = (preds > base_th).astype('uint8')
        cls_preds = get_preds(outputs, cfg['num_classes'])
        print("base:", np.unique(cls_preds, return_counts=True)[1])
        if cfg['phase'] == 'test':
            score = cohen_kappa_score(y_test, cls_preds, weights="quadratic")
            acc = accuracy_score(y_test, cls_preds)
            print(f'base qwk: {score}, acc: {acc}')
            cm = ConfusionMatrix(y_test, cls_preds.flatten())
            cls_f1 = sanitize(cm.class_stat["F1"])
            print(f'F1: {cls_f1} \n')
        """
        print(f"Best thresholds: {best_thresholds}")
        pred1 = predict(preds, best_thresholds)
        print("best:", np.unique(pred1, return_counts=True)[1])
        """
        mat_to_save = [preds, base_th]
        np.save(os.path.join(npy_folder, f"{predict_on}_ckpt{epoch}.npy"), mat_to_save)
        print("Predictions saved!")


"""
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
"""

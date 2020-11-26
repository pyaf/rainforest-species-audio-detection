import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from models import Model, get_model
from utils import *
from image_utils import *

# from submission import get_best_threshold


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--model_folder_path",
        dest="model_folder_path",
        help="relative path to the folder where model checkpoints are saved",
    )
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="resnext101_32x4d",
    )
    return parser


class Dataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.TTA = albumentations.Compose(
            [
                # albumentations.RandomRotate90(p=1),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),
            ]
        )
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        # image = load_image(path, size)
        # image = load_ben_gray(path)
        image = load_ben_color(path, size=self.size, crop=True)

        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def get_predictions(model, testset, tta):
    """return all predictions on testset in a list"""
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if tta:
            # images.shape [n, 3, 96, 96] where n is num of 1+tta
            for images in batch:
                preds = model(images.to(device))  # [n, num_classes]
                predictions.append(preds.mean(dim=0).detach().tolist())
        else:
            preds = model(batch[:, 0].to(device))
            preds = preds.detach().tolist()  # [1]
            predictions.extend(preds)

    return np.array(predictions)


def get_load_model(model_name, ckpt_path, num_classes):
    model = get_model(model_name, num_classes, pretrained=None)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    epoch = state["epoch"]
    model.load_state_dict(state["state_dict"])

    best_thresholds = state["best_thresholds"]
    model.to(device)
    model.eval()
    return model, best_thresholds


def get_model_name_fold(model_folder_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/
    model_folder = model_folder_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2]  # fold0
    fold = fold.split("fold")[-1]  # 0
    return model_name, int(fold)


if __name__ == "__main__":
    """
    Uses a list of ckpts, predicts on whole train set, averages the predictions and finds optimized thresholds based on train qwk
    """
    model_name = "efficientnet-b5"
    ckpt_path_list = [
        # "weights/19-7_efficientnet-b5_fold0_bgccpold/ckpt20.pth",
        # "weights/19-7_efficientnet-b5_fold1_bgccpold/ckpt10.pth",
        # "weights/19-7_efficientnet-b5_fold2_bgccpold/ckpt30.pth",
        # "weights/19-7_efficientnet-b5_fold3_bgccpold/ckpt15.pth"
        "weights/21-7_efficientnet-b5_fold1_bgccpo300/ckpt20.pth"
    ]

    # folds = [0, 1, 2, 3] # for extracting val sets, used for thr optimization
    folds = [1]
    sample_submission_path = "data/train.csv"

    tta = 4  # number of augs in tta
    total_folds = 7

    root = f"data/train_images/"
    size = 300
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    use_cuda = True
    num_classes = 1
    num_workers = 8
    batch_size = 16
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    df = pd.read_csv(sample_submission_path)

    # kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    # index_list = list(kfold.split(df["id_code"], df["diagnosis"]))

    # val_idx = []
    # for fold in folds:
    #    val_idx.extend(index_list[fold][1])

    # df = df.iloc[val_idx]

    dataset = DataLoader(
        Dataset(root, df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )
    print(f"len dataset: {len(dataset)}")

    # generate predictions using all models
    all_predictions = []
    for idx, ckpt in enumerate(ckpt_path_list):
        print("model: %s" % ckpt)
        model, val_best_th = get_load_model(model_name, ckpt, num_classes)
        predictions = get_predictions(model, dataset, tta)
        all_predictions.append(predictions)
        # break

    predictions = np.mean(all_predictions, axis=0).flatten()

    # optimize thresholds on training set
    targets = df["diagnosis"].values
    initial_thresholds = [0.5, 1.5, 2.5, 3.5]
    simplex = scipy.optimize.minimize(
        compute_score_inv,
        initial_thresholds,
        args=(predictions, targets),
        method="nelder-mead",
    )
    best_thresholds = simplex["x"]
    print("Best thresholds: %s" % best_thresholds)

    # predictions using best_thresholds
    preds = predict(predictions, best_thresholds)

    qwk = cohen_kappa_score(preds, targets, weights="quadratic")
    print(f"Train qwk score: {qwk}")

    cm = ConfusionMatrix(targets, preds)
    print(cm.print_normalized_matrix())

    # for further analysis.
    pdb.set_trace()

    # now use the best_threshold on test data to generate predictions

    df = pd.read_csv("data/sample_submission.csv")
    root = f"data/test_images/"
    testset = DataLoader(
        Dataset(root, df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )
    # generate predictions using all models
    base_thresholds = np.array([0.5, 1.5, 2.5, 3.5])
    all_predictions = []
    for idx, ckpt in enumerate(ckpt_path_list):
        print("model: %s" % ckpt)
        model, val_best_th = get_load_model(model_name, ckpt, num_classes)
        predictions = get_predictions(model, testset, tta)
        preds = predict(predictions, best_thresholds)
        print(np.unique(preds, return_counts=True))
        all_predictions.append(predictions)
        # break
    predictions = np.mean(all_predictions, axis=0).flatten()
    preds = predict(predictions, best_thresholds)
    print(np.unique(preds, return_counts=True))

    pdb.set_trace()

"""
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
"""

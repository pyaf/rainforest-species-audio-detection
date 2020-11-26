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
from models import get_model
from utils import *
from image_utils import *
from augmentations import *


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-c", "--ckpt_path", dest="ckpt_path", help="Checkpoint to use")
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="test",
    )
    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.TTA = get_test_transforms(size, mean, std)
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # path = os.path.join(self.root, fname + ".png")
        # image = load_ben_color(path, size=self.size, crop=True)
        path = os.path.join(self.root, fname + ".npy")
        image = np.load(path)

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


def get_model_name_fold(ckpt_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/ckpt12.pth
    model_folder = ckpt_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2]  # fold0
    fold = fold.split("f")[-1]  # 0
    return model_name, int(fold)


if __name__ == "__main__":
    """
    use given ckpt to generate final predictions using the corresponding best thresholds.
    """
    parser = get_parser()
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    predict_on = args.predict_on
    model_name, fold = get_model_name_fold(ckpt_path)
    model_name = "efficientnet-b5"
    if predict_on == "test":
        sample_submission_path = "data/sample_submission.csv"
    else:
        sample_submission_path = "data/train.csv"

    sub_path = ckpt_path.replace(".pth", f"{predict_on}.csv")
    tta = 4  # number of augs in tta

    # root = f"data/{predict_on}_images/"
    root = "data/npy_files/bgcc456"
    size = 456
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    use_cuda = True
    num_classes = 1
    num_workers = 2
    batch_size = 4
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(model_name, num_classes, pretrained=None)
    model.to(device)
    model.eval()

    print(f"Using {ckpt_path}")
    print(f"Predicting on: {predict_on} set")
    print(f"Root: {root}")
    print(f"Using tta: {tta}\n")
    print(f"batch_size: {batch_size}")

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    best_thresholds = state["best_thresholds"]
    base_thresholds = [0.5, 1.5, 2.5, 3.5]
    # best_thresholds = base_thresholds
    print("best_thresholds:", best_thresholds)

    predictions = get_predictions(model, testset, tta)
    best_preds = predict(predictions, np.array(best_thresholds))
    base_preds = predict(predictions, np.array(base_thresholds))
    print("best:", np.unique(best_preds, return_counts=True)[1])
    print("base:", np.unique(base_preds, return_counts=True)[1])

    pdb.set_trace()
    df.loc[:, "diagnosis"] = base_preds
    print(f"Saving predictions at {sub_path}")
    df.to_csv(sub_path, index=False)
    print("Predictions saved!")

    # print(np.unique(predict(predictions, np.array([0.5, 1.5, 2.5, 3.5])), return_counts=True)[1])

"""
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
"""

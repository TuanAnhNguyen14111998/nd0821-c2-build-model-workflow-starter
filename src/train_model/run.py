#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
from ast import arg
import logging
from os import path
import mlflow
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import densenet121

import mlflow
import json

import pandas as pd
import numpy as np
from mlflow.models import infer_signature
import albumentations as albu

import sys
sys.path.append("../../")

from library.TedAI.tedai import *
from library.cxr_code.models.convnext import convnext_tiny,  LayerNorm


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainTransforms:
    def __init__(self, img_size):
        self.aug_func = partial(train_transforms, img_size=img_size)

    def __call__(self, x):
        return self.aug_func(x)


def train_transforms(x, img_size):
    albu_tfs = albu.Compose([
        albu.RandomBrightnessContrast(0.4, 0.4),
        albu.ShiftScaleRotate(0.1, (-0.09, 0.09), 25, always_apply=True),
        albu.SmallestMaxSize(img_size, always_apply=True),
        albu.RandomCrop(img_size, img_size),
        albu.Perspective(0.1, p=0.4),
        albu.HorizontalFlip(p=0.5),   
    ])
    torch_tfs = Compose([
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Cutout(10, img_size//10),
    ])
    
    x = albu_tfs(image=x)['image']
    x = torch_tfs(x)
    return x


def f1_NDI(preds, targets):
    return f1_score(preds, targets, sigmoid=True, average='binary', idx=1)


class BCELogLoss(nn.Module):
    def __init__(self, device, smooth=False):
        super(BCELogLoss, self).__init__()
        self.device = device
        # label smoothing
        self.smooth = smooth
    def forward(self, output, target):
        # Label smoothing
        if self.smooth:
            eps = np.random.uniform(0.01, 0.05)
            target = (1-eps)*target + eps / target.size()[1]
        return F.binary_cross_entropy_with_logits(output, target)

def go(args):

    with mlflow.start_run(run_name="train_model") as mlrun:

        # Get the Random Forest configuration and update W&B
        with open(args.model_config) as fp:
            model_config = json.load(fp)
        
        storage_id = args.storage_id
        data_train_path = f"../../data/information/{storage_id}/dataset/train.csv"
        data_test_path = f"../../data/information/{storage_id}/dataset/train.csv"

        train = pd.read_csv(data_train_path).sample(10, random_state=42).reset_index(drop=True)
        test = pd.read_csv(data_test_path).sample(10, random_state=42).reset_index(drop=True)

        train_ds = create_dataset(TedImageDataset, df=train, label_cols_list=model_config["classes"].split(","))
        test_ds = create_dataset(TedImageDataset, df=test, label_cols_list=model_config["classes"].split(","))

        valid_transforms = lambda img_size: Compose([
            ToPILImage(), Resize(int(img_size*1.05)), 
            CenterCrop((img_size, img_size)), ToTensor(), 
            Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
        ])

        data = TedData(data_path=args.folder_path, 
               ds_class=(train_ds,test_ds), 
               transforms=(TrainTransforms, valid_transforms), 
               img_size=args.image_size, bs=model_config["batch_size"], 
               n_workers=model_config["n_workers"])
        
        data.show_batch(mode='train', path_save=args.output_artifact + "train.png")
        data.show_batch(mode='valid', path_save=args.output_artifact + "valid.png")

        CLASSES = model_config["classes"].split(",")

        fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(), 
                nn.Dropout(0.25), 
                nn.Linear(1024, 
                len(CLASSES), 
                bias=False)
            )
        model = TedModel(arch=densenet121(True).features[:-1], hidden_size=1024, num_classes=len(CLASSES), head=fc)

        criterion = BCELogLoss(device=device, smooth=False)

        opt_func = partial(torch.optim.SGD, momentum=0.97, nesterov=True)

        learn = TedLearner(data=data, model=model, 
            model_path=args.output_artifact,
            opt_func=opt_func,
            loss_func=criterion, 
            metrics=[f1_NDI], 
            show_imgs=True, log=args.output_artifact + f"/{model_config['model_name']}.html"
        )

        config_model_save(learn.recorder, name=model_config['model_name'], mode='improve', monitor='f1_NDI') 
        learn.fit_one_cycle(
            model_config['epoch_num'], 
            (float(model_config['min_lr']), float(model_config['max_lr'])), 
            name=model_config['model_name']
        )

        mlflow.log_artifacts(args.output_artifact, artifact_path="model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--folder_path",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--storage_id",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--model_config",
        help="Model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)

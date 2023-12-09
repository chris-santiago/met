import inspect
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset
from torchmetrics.classification import AUROC, Accuracy

import met.constants
import met.data
from met.models.met import MET
from met.utils import mask_tensor_2d

constants = met.constants.Constants()
BINARY = "binary"
MULTICLASS = "multiclass"


@dataclass()
class Model:
    module: Union[type, pl.LightningModule]
    ckpt_path: str
    device: str = "cpu"

    def __post_init__(self):
        device = torch.device(self.device)
        if inspect.isclass(self.module):
            self.module = self.module.load_from_checkpoint(self.ckpt_path, map_location=device)


def preprocess_data(
    model: Union[Model, pl.LightningModule], data: Union[TensorDataset, torch.Tensor]
):
    if isinstance(model, Model):
        model = model.module

    if isinstance(data, torch.Tensor):
        x, idx_train, _ = mask_tensor_2d(data.flatten(1), pct_mask=0)
        x = model.encode(x, idx_train).flatten(1)
        return x
    else:
        x, y = data.tensors
        x, idx_train, _ = mask_tensor_2d(x.flatten(1), pct_mask=0)
        x = model.encode(x, idx_train).flatten(1)
        return x, y


def evaluate_classifier(
    cls: BaseEstimator,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
):
    n_classes = len(y_train.unique())
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    if n_classes > 2:
        task = MULTICLASS
        # Correct for issues where labels start at 1 vice 0...this affects multiclass metrics
        # TODO this should be handled in dataset / preprocessing...but model is already trained
        if y_train.min() != 0:
            y_train -= 1
        if y_test.min() != 0:
            y_test -= 1
    else:
        task = BINARY

    cls.fit(x_train, y_train)
    labels = torch.tensor(cls.predict(x_test))

    acc = Accuracy(task=task, num_classes=n_classes if n_classes > 2 else None)
    auc = AUROC(task=task, num_classes=n_classes if n_classes > 2 else None)

    acc_score = acc(labels, y_test)
    if task == MULTICLASS:
        labels_ohe = F.one_hot(labels).float()
        auc_score = auc(labels_ohe, y_test)
    else:
        auc_score = auc(labels, y_test)

    return acc_score, auc_score


def format_results(acc_score, auc_score, ckpt_path):
    # pull out .item() for metrics tensors as tensors are not json serializable
    return {
        "metrics": {
            "acc": round(acc_score.item(), 4),
            "auc": round(auc_score.item(), 4),
        },
        "ckpt": str(ckpt_path),
    }


def to_json(results: Dict, filepath: pathlib.Path):
    if filepath.exists():
        with open(filepath, "r") as fp:
            res = json.load(fp)
        res.append(results)
    else:
        res = [results]
    with open(filepath, "w") as fp:
        json.dump(res, fp, indent=2)


def main(ckpt_path: str):
    ckpt_path = str(constants.OUTPUTS.joinpath(ckpt_path))
    model = Model(MET, ckpt_path)
    x_train, y_train = preprocess_data(
        model, met.data.get_covertype_dataset(train=True, subset=0.25)
    )
    x_test, y_test = preprocess_data(model, met.data.get_covertype_dataset(train=False))
    cls = LogisticRegression(max_iter=3, C=0.1, tol=1e-4, solver="saga", verbose=1, n_jobs=4)
    acc_score, auc_score = evaluate_classifier(cls, x_train, y_train, x_test, y_test)
    results = format_results(acc_score, auc_score, ckpt_path)
    to_json(results, constants.OUTPUTS.joinpath("results.json"))


if __name__ == "__main__":
    main(sys.argv[1])

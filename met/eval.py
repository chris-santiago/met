import inspect
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset
from torchmetrics.classification import AUROC, Accuracy

import met.constants
from met.data import get_income_dataset
from met.models.met import MET
from met.utils import mask_tensor_2d

constants = met.constants.Constants()
BINARY = "binary"
MULTICLASS = "multiclass"


@dataclass()
class Model:
    module: pl.LightningModule
    ckpt_path: str
    device: str = "cpu"

    def __post_init__(self):
        device = torch.device(self.device)
        if inspect.isclass(self.module):
            self.module = self.module.load_from_checkpoint(self.ckpt_path, map_location=device)


def preprocess_data(model: Model, data: TensorDataset):
    x, y = data.tensors
    x, idx_train, _ = mask_tensor_2d(x.flatten(1), pct_mask=0)
    x = model.module.encode(x, idx_train).flatten(1)
    return x, y


def evaluate_classifier(
    model: Model,
    cls: BaseEstimator,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
):
    n_classes = len(np.unique(y_train))
    if n_classes > 2:
        task = MULTICLASS
    else:
        task = BINARY
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    cls.fit(x_train, y_train.ravel())
    labels = torch.tensor(cls.predict(x_test))
    if task == MULTICLASS:
        labels = F.one_hot(labels).float()

    acc = Accuracy(task=task, num_classes=n_classes if n_classes > 2 else None)
    auc = AUROC(task=task, num_classes=n_classes if n_classes > 2 else None)

    # pull out .item() for metrics tensors as tensors are not json serializable
    return {
        "metrics": {
            "acc": round(acc(labels, y_test).item(), 4),
            "auc": round(auc(labels, y_test).item(), 4),
        },
        "ckpt": str(model.ckpt_path),
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
    x_train, y_train = preprocess_data(model, get_income_dataset(train=True))
    x_test, y_test = preprocess_data(model, get_income_dataset(train=False))
    cls = LogisticRegression(max_iter=3000)
    results = evaluate_classifier(model, cls, x_train, y_train, x_test, y_test)
    to_json(results, constants.OUTPUTS.joinpath("results.json"))


if __name__ == "__main__":
    main(sys.argv[1])

"""
>>> hydra.initialize(config_path='met/conf', version_base="1.3")
>>> cfg = hydra.compose(config_name='comps')
"""
import json
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torchmetrics.classification import AUROC, Accuracy

from met.constants import Constants
from met.data import get_income_dataset
from met.utils import mask_tensor_2d

constants = Constants()


def encoder_helper(encoder: pl.LightningModule, x):
    """For modules without .encode() method."""
    x, idx, _ = mask_tensor_2d(x, pct_mask=0)
    inputs = encoder.embed_inputs(x, idx)
    z = encoder.transformer.encoder(inputs)
    return z.flatten(1).detach()


def encoder_results(encoder: Optional[pl.LightningModule] = None, seed: Optional[int] = None):
    ds = get_income_dataset(train=False)

    train_size = [10, 100, 1000, 2000, 4000, 8000]
    acc_scores = []
    auc_scores = []
    for i in train_size:
        x_train, x_test, y_train, y_test = train_test_split(
            ds.tensors[0], ds.tensors[1], train_size=i, stratify=ds.tensors[1], random_state=seed
        )
        if encoder:
            if hasattr(encoder, "encode"):
                x_train, idx_train, _ = mask_tensor_2d(x_train.flatten(1), pct_mask=0)
                x_test, idx_test, _ = mask_tensor_2d(x_test.flatten(1), pct_mask=0)
                x_train = encoder.encode(x_train, idx_train).flatten(1).numpy()
                x_test = encoder.encode(x_test, idx_test).flatten(1).numpy()
            else:
                x_train = encoder_helper(encoder, x_train).numpy()
                x_test = encoder_helper(encoder, x_test).numpy()

        else:
            x_train = x_train.flatten(1).numpy()
            x_test = x_test.flatten(1).numpy()

        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_train, y_train.ravel())
        labels = lr.predict(x_test)

        num_classes = len(ds.tensors[1].unique())
        acc = Accuracy(task="binary")
        auc = AUROC(task="binary", num_classes=num_classes)

        acc_scores.append(round(acc(torch.tensor(labels), y_test.ravel()).item(), 4))
        auc_scores.append(round(auc(torch.tensor(labels), y_test.ravel()).item(), 4))

    return {"train_size": train_size, "acc": acc_scores, "auc": auc_scores}


@hydra.main(config_path="conf", config_name="comps", version_base="1.3")
def main(cfg):
    results = {"NoPretraining": encoder_results(seed=cfg.seed)}
    for model in cfg.models:
        # module = hydra.utils.instantiate(cfg.models[model].module)
        # Latest Lightning version loads checkpoints from class, not instance
        cls = hydra.utils.get_class(cfg.models[model].cls)
        ckpt_path = constants.OUTPUTS.joinpath(cfg.models[model].ckpt_path)
        encoder = cls.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        results[cfg.models[model].name] = encoder_results(encoder, seed=cfg.seed)

    with open(constants.OUTPUTS.joinpath("comps.json"), "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()

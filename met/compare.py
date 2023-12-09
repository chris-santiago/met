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
from torch.utils.data import TensorDataset

from met.constants import Constants
from met.eval import evaluate_classifier, preprocess_data
from met.utils import mask_tensor_2d

constants = Constants()


def encoder_helper(encoder: pl.LightningModule, x):
    """For modules without .encode() method."""
    x, idx, _ = mask_tensor_2d(x, pct_mask=0)
    inputs = encoder.embed_inputs(x, idx)
    z = encoder.transformer.encoder(inputs)
    return z.flatten(1).detach()


def encoder_results(
    dataset: TensorDataset, encoder: Optional[pl.LightningModule] = None, seed: Optional[int] = None
):
    train_size = [0.1, 0.25, 0.5, 0.8]
    acc_scores = []
    auc_scores = []
    for i in train_size:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.tensors[0],
            dataset.tensors[1],
            train_size=i,
            stratify=dataset.tensors[1],
            random_state=seed,
        )
        if encoder:
            x_train = preprocess_data(encoder, x_train)
            x_test = preprocess_data(encoder, x_test)
        else:
            x_train = x_train.flatten(1).numpy()
            x_test = x_test.flatten(1).numpy()

        cls = LogisticRegression(max_iter=300, C=0.1, solver="saga")
        acc_score, auc_score = evaluate_classifier(cls, x_train, y_train, x_test, y_test)

        acc_scores.append(round(acc_score.item(), 4))
        auc_scores.append(round(auc_score.item(), 4))

    return {"train_size": train_size, "acc": acc_scores, "auc": auc_scores}


@hydra.main(config_path="conf", config_name="comps", version_base="1.3")
def main(cfg):
    data = hydra.utils.instantiate(cfg.comps.dataset)
    results = {"NoPretraining": encoder_results(data, seed=cfg.seed)}
    for model in cfg.comps.models:
        # Latest Lightning version loads checkpoints from class, not instance
        cls = hydra.utils.get_class(cfg.comps.models[model].cls)
        ckpt_path = constants.OUTPUTS.joinpath(cfg.comps.models[model].ckpt_path)
        encoder = cls.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        results[cfg.comps.models[model].name] = encoder_results(data, encoder, seed=cfg.seed)

    save_dir = constants.OUTPUTS.joinpath("comps")
    save_dir.mkdir(exist_ok=True)
    with open(save_dir.joinpath(f"{cfg.comps.name}.json"), "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()

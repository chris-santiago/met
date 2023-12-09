from typing import Optional

import pandas as pd
import torch
from feature_engine.encoding import OrdinalEncoder

import met.constants

constants = met.constants.Constants()
constants.DATA.mkdir(exist_ok=True)


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]


def random_choice(x, pct_mask):
    n = len(x)
    n_choice = int((pct_mask * 100 * n) // 100)
    idx = torch.randperm(n)
    return idx[:n_choice]


def mask_tensor_1d(x, pct_mask: float = 0.7):
    n = len(x)
    n_masked = int((pct_mask * 100 * n) // 100)
    idx = torch.randperm(n)
    masked_idx, _ = idx[:n_masked].sort()
    unmasked_idx, _ = idx[n_masked:].sort()
    unmasked_x = torch.zeros_like(unmasked_idx, dtype=torch.float)
    unmasked_x += x[unmasked_idx]
    return unmasked_x, unmasked_idx, masked_idx


def mask_tensor_2d(x, pct_mask: float = 0.7, dim=1):
    n_row, n_col = x.shape
    n_masked = int((pct_mask * 100 * n_col) // 100)
    idx = torch.stack([torch.randperm(n_col) for _ in range(n_row)])
    masked_idx, _ = idx[:, :n_masked].sort(dim=dim)
    unmasked_idx, _ = idx[:, n_masked:].sort(dim=dim)
    unmasked_x = torch.zeros(n_row, unmasked_idx.shape[dim])
    for i in range(n_row):
        unmasked_x[i] += x[i][unmasked_idx[i]]
    return unmasked_x, unmasked_idx, masked_idx


def prep_adult_income(dir_path: Optional[str] = None):
    if dir_path is None:
        dir_path = constants.DATA.joinpath("adult")

    train = pd.read_csv(dir_path.joinpath("adult.data"), header=None)
    test = pd.read_csv(dir_path.joinpath("adult.test"), header=None)

    cats = train.select_dtypes(include="object").columns
    for df in [train, test]:
        df.loc[:, cats] = df.loc[:, cats].map(str.strip)

    od = OrdinalEncoder(encoding_method="arbitrary")
    data = {
        "x_train": od.fit_transform(train.iloc[:, :-1]),
        "x_test": od.transform(test.iloc[:, :-1]),
        "y_train": train.iloc[:, -1].str.contains(">50K").astype(int),
        "y_test": test.iloc[:, -1].str.contains(">50K").astype(int),
    }

    for name, ds in data.items():
        ds.to_csv(dir_path.joinpath(f"{name}.csv"), index=False, header=False)
    return data

from typing import Callable, Optional

import pandas as pd
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T
from sklearn.preprocessing import MinMaxScaler

import met.constants
import met.utils

constants = met.constants.Constants()

# Transform to make tensor, scale, and flatten
make_tabular = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])


def get_mnist_dataset(train: bool = True, transform: Callable = make_tabular):
    return torchvision.datasets.MNIST(
        constants.DATA, train=train, download=True, transform=transform
    )


def get_income_dataset(train: bool = True, transform: Optional[Callable] = None):
    ds = {True: "train", False: "test"}
    sc = MinMaxScaler()
    x = pd.read_csv(constants.DATA.joinpath("adult", f"x_{ds[train]}.csv"), header=None)
    y = pd.read_csv(constants.DATA.joinpath("adult", f"y_{ds[train]}.csv"), header=None)
    return torch.utils.data.TensorDataset(
        torch.tensor(sc.fit_transform(x), dtype=torch.float32), torch.tensor(y.values)
    )


def scale_mnist(x: torch.Tensor) -> torch.Tensor:
    return x / 255


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, outputs = self.dataset.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, outputs


class METDataset(MnistDataset):
    def __init__(self, dataset, transform=None, pct_mask: float = 0.7):
        self.dataset = dataset
        self.transform = transform
        self.pct_mask = pct_mask
        super().__init__(dataset, transform)

    def __getitem__(self, idx):
        inputs, outputs = self.dataset.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        unmasked_x, unmasked_idx, masked_idx = met.utils.mask_tensor_1d(inputs, self.pct_mask)
        masked_x = torch.ones_like(masked_idx)
        return unmasked_x, unmasked_idx, masked_x, masked_idx, inputs, outputs

from typing import Callable

import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T

import met.constants

constants = met.constants.Constants()

# Transform to make tensor, scale, and flatten
make_tabular = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])


def get_mnist_dataset(train: bool = True, transform: Callable = make_tabular):
    return torchvision.datasets.MNIST(
        constants.DATA, train=train, download=True, transform=transform
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

import torch

import met.constants

constants = met.constants.Constants()
constants.DATA.mkdir(exist_ok=True)


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]


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

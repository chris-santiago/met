from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        loss_func: nn.Module = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()
        self.loss_func = loss_func
        self.optim = optim
        self.scheduler = scheduler
        self.save_hyperparameters(ignore=["loss_func"])

    def configure_optimizers(self) -> Any:
        optim = self.optim(self.parameters()) if self.optim else torch.optim.Adam(self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optim)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train-loss",
                    "interval": "epoch",
                },
            }
        return optim  # torch.optim.Adam(self.parameters())

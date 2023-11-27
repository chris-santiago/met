import typing as T

import hydra
import omegaconf
import pytorch_lightning as pl


def instantiate_callbacks(callbacks_cfg: omegaconf.DictConfig) -> T.List[pl.Callback]:
    """Instantiates callbacks from config."""

    callbacks: T.List[pl.Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, omegaconf.DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, omegaconf.DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

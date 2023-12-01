from typing import Optional

import torch
from torch import nn as nn

from met.models.base import BaseModule


def mask_tensor(x, pct_mask: float = 0.7, dim=1):
    n_row, n_col = x.shape
    n_masked = int((pct_mask * 100 * n_col) // 100)
    idx = torch.stack([torch.randperm(n_col) for _ in range(n_row)])
    masked_idx, _ = idx[:, :n_masked].sort(dim=dim)
    unmasked_idx, _ = idx[:, n_masked:].sort(dim=dim)
    unmasked_x = torch.zeros(n_row, unmasked_idx.shape[dim])
    for i in range(n_row):
        unmasked_x += x[i][unmasked_idx[i]]
    return unmasked_x, unmasked_idx, masked_idx


class MET(BaseModule):
    def __init__(
        self,
        num_embeddings: int = 784,
        embedding_dim: int = 64,
        p_mask: float = 0.70,
        n_head: int = 1,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        loss_func: nn.Module = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(loss_func, optim, scheduler)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.p_mask = p_mask
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Subtract 1 from desired embedding dim to account for token
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim - 1
        )
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.transformer_head = nn.Linear(embedding_dim, 1)

    def training_step(self, batch, idx):
        original = batch[0]
        # inputs: (batch, cols)
        # mask: (batch, rows, mask_cols)
        # embeds: (mask_cols, embed_dim)
        # new inputs: (batch, mask_cols, embed_dim)
        x, unmasked_idx, masked_idx = mask_tensor(batch[0], pct_mask=self.p_mask)

        # Transformer inputs are concat of original tokens and position embed
        unmasked_embd = self.embedding(unmasked_idx)
        unmasked_inputs = torch.concat([x.unsqueeze(-1), unmasked_embd], dim=-1)

        # Need a constant tensor of masked inputs to learn embed params
        masked_embd = self.embedding(masked_idx)
        masked_inputs = torch.concat(
            [torch.ones_like(masked_idx).unsqueeze(-1), masked_embd], dim=-1
        )

        # Recovering embeddings for all original features/cols (784 in MNIST example)
        all_embd = torch.concat([unmasked_inputs, masked_inputs], dim=1)

        outputs = self.transformer(unmasked_inputs, all_embd)
        recon = self.transformer_head(outputs).squeeze()
        recon_loss = self.loss_func(original, recon)
        self.log("recon-loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        # TODO adversarial loss (make for loop)
        h = torch.normal(torch.zeros_like(x), torch.ones_like(x)) / torch.sqrt(
            torch.tensor(x.shape[1])
        )
        # for _ in range(adver_steps):
        perturbed = x + h
        perturbed_inputs = torch.concat([perturbed.unsqueeze(-1), unmasked_embd], dim=-1)
        perturbed_outputs = self.transformer(perturbed_inputs, all_embd)
        perturbed_recon = self.transformer_head(perturbed_outputs).squeeze()
        perturbed_loss = self.loss_func(original, perturbed_recon)
        # TODO gradient ascent...but how? Separate optimizer? Can I just add gradient?
        # TODO do I need a separate optimzer? (https://discuss.pytorch.org/t/pytorch-equivalant-of-tensorflow-gradienttape/74915)
        # TODO https://discuss.pytorch.org/t/gradient-ascent-and-gradient-modification-modifying-optimizer-instead-of-grad-weight/62777/2
        # TODO https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#gradient-accumulation
        h += perturbed_loss.grad
        adver_loss = self.loss_func(original, perturbed_recon)
        self.log("adver-loss", adver_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        total_loss = recon_loss + adver_loss
        self.log("total-loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        metrics = {
            "reconstruction-loss": recon_loss,
            "adversarial-loss": adver_loss,
            "train-total-loss": total_loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

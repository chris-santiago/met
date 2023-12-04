from typing import Optional

import numpy as np
import torch
from torch import nn as nn

from met.models.base import BaseModule


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
        adver_steps: int = 2,
        lr_perturb: float = 1e-4,
        eps: int = 12,
        lam: float = 1.0,
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
        self.adver_steps = adver_steps
        self.lr_perturb = lr_perturb
        self.eps = eps
        self.lam = lam

        self.automatic_optimization = False

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
            batch_first=True,
        )
        self.transformer_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Flatten())
        # self.transformer_head = nn.Linear(embedding_dim, 1)

    def embed_inputs(self, x, idx):
        # Transformer inputs are concat of original tokens and position embed
        embd = self.embedding(idx)
        return torch.concat([x.unsqueeze(-1), embd], dim=-1)

    def forward(self, unmasked_x, unmasked_idx, masked_x, masked_idx):
        unmasked_inputs = self.embed_inputs(unmasked_x, unmasked_idx)
        masked_inputs = self.embed_inputs(masked_x, masked_idx)

        # Recovering embeddings for all original features/cols (784 in MNIST example)
        all_embd = torch.concat([unmasked_inputs, masked_inputs], dim=1)
        outputs = self.transformer(unmasked_inputs, all_embd)
        x_hat = self.transformer_head(outputs)
        return x_hat

    def encode(self, x, idx):
        with torch.no_grad():
            inputs = self.embed_inputs(x, idx)
            return self.transformer.encoder(inputs)

    def adversarial_step(self, unmasked_x, unmasked_idx, masked_x, masked_idx, original):
        opt = self.optimizers()
        unmasked_x.retain_grad()
        perturbed_recon = self(unmasked_x, unmasked_idx, masked_x, masked_idx)
        perturbed_loss = -self.loss_func(original, perturbed_recon)  # grad ascent
        opt.zero_grad()
        self.manual_backward(perturbed_loss, retain_graph=True)
        # !! Interesting-- if this is enabled it causes in-place errors w/grad calcs !!
        # I believe it modifies the `recon_loss` grad that's done before this
        # opt.step()

        # Constrain h
        h = unmasked_x + self.lr_perturb * (unmasked_x.grad / torch.norm(unmasked_x.grad))
        alpha = (torch.norm(h) * int(torch.norm(h) < self.eps)) + (
            self.eps * int(torch.norm(h) >= self.eps)
        )
        h_adj = alpha * (h / torch.norm(h))
        return unmasked_x + h_adj

    def training_step(self, batch, idx):
        opt = self.optimizers()
        # inputs: (batch, cols)
        # mask: (batch, rows, mask_cols)
        # embeds: (mask_cols, embed_dim)
        # new inputs: (batch, mask_cols, embed_dim)
        unmasked_x, unmasked_idx, masked_x, masked_idx, original, _ = batch

        # standard reconstruction loss
        recon = self(unmasked_x, unmasked_idx, masked_x, masked_idx)
        recon_loss = self.loss_func(original, recon)
        self.log("recon-loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        # adversarial loss
        h = torch.normal(0, 1, size=unmasked_x.shape, device=self.device, requires_grad=True)
        perturbed_x = (unmasked_x.clone() + h) / np.sqrt(original.shape[-1])
        for i in range(self.adver_steps):
            h = self.adversarial_step(perturbed_x, unmasked_idx, masked_x, masked_idx, original)
            perturbed_x = h.clone()
        adv_recon = self(perturbed_x, unmasked_idx, masked_x, masked_idx)
        adv_loss = self.loss_func(original, adv_recon)
        self.log("adv-loss", adv_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        total_loss = recon_loss + adv_loss * self.lam
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        self.log("total-loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        metrics = {
            "reconstruction-loss": recon_loss,
            "adversarial-loss": adv_loss,
            "train-loss": total_loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # return total_loss

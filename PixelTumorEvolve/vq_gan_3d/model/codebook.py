""" Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.utils import shift_dim


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=0.05):
        super().__init__()

        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', torch.zeros(n_codes, embedding_dim))

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim

        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.training_steps = 0

        # EMA decay (stable default for spherical VQ)
        self.ema_decay = 0.99

    # -----------------------------
    # INIT HELPERS
    # -----------------------------
    def _tile(self, x):
        d, c = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(c)

            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std

        return F.normalize(x, dim=1)

    # -----------------------------
    # FORWARD
    # -----------------------------
    def forward(self, z):
        # z: [b, c, t, h, w]

        # 1. SINGLE normalization (critical fix)
        z = F.normalize(z, p=2, dim=1)

        if self._need_init and self.training:
            self._init_embeddings(z)

        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)  # [N, C]

        # 2. cosine codebook (normalized on the fly)
        emb_norm = F.normalize(self.embeddings, dim=1)

        # cosine similarity (stable spherical VQ)
        with torch.no_grad():
            logits = flat_inputs @ emb_norm.t()
            encoding_indices = torch.argmax(logits, dim=1)

        enc_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)

        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        # quantize
        embeddings = F.embedding(encoding_indices, emb_norm)
        embeddings = shift_dim(embeddings, -1, 1)

        # -----------------------------
        # COMMITMENT LOSS (FIXED)
        # -----------------------------
        # IMPORTANT: NO extra scaling by C
        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # -----------------------------
        # EMA UPDATE (STANDARD SPHERICAL VQ)
        # -----------------------------
        if self.training:
            with torch.no_grad():
                n_total = enc_onehot.sum(dim=0)  # [K]
                encode_sum = enc_onehot.t() @ flat_inputs  # [K, C]

                if dist.is_initialized():
                    dist.all_reduce(n_total)
                    dist.all_reduce(encode_sum)

                # stable EMA update
                self.N.mul_(self.ema_decay).add_(n_total, alpha=1 - self.ema_decay)
                self.z_avg.mul_(self.ema_decay).add_(encode_sum, alpha=1 - self.ema_decay)

                denom = self.N.unsqueeze(1).clamp(min=1e-5)
                new_emb = self.z_avg / denom

                # re-normalize for spherical constraint
                new_emb = F.normalize(new_emb, dim=1)

                self.embeddings.data.copy_(new_emb)

                # -----------------------------
                # RANDOM RESTART (kept but stabilized)
                # -----------------------------
                if not self.no_random_restart:
                    y = self._tile(flat_inputs)
                    idx = torch.randint(
                        y.shape[0],
                        (self.n_codes,),
                        device=y.device
                    )

                    rand_codes = y[idx]

                    if dist.is_initialized():
                        dist.broadcast(rand_codes, 0)

                    # usage is 1D: [n_codes]
                    usage = (self.N >= self.restart_thres).float()

                    # 2D masks for embeddings and z_avg
                    keep_2d = usage.unsqueeze(1)
                    reset_2d = 1.0 - keep_2d

                    # 1D masks for N
                    keep_1d = usage
                    reset_1d = 1.0 - usage

                    self.embeddings.data.copy_(
                        self.embeddings.data * keep_2d + rand_codes * reset_2d
                    )

                    self.z_avg.data.copy_(
                        self.z_avg.data * keep_2d + rand_codes * reset_2d
                    )

                    self.N.data.copy_(
                        self.N.data * keep_1d + torch.ones_like(self.N) * reset_1d
                    )

        # -----------------------------
        # STE (unchanged, correct)
        # -----------------------------
        embeddings_st = (embeddings - z).detach() + z

        # perplexity
        avg_probs = torch.mean(enc_onehot.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    # -----------------------------
    # INIT
    # -----------------------------
    def _init_embeddings(self, z):
        flat = shift_dim(z, 1, -1).flatten(end_dim=-2)

        y = self._tile(flat)
        idx = torch.randint(
            y.shape[0],
            (self.n_codes,),
            device=y.device
        )

        init = y[idx]
        init = F.normalize(init, dim=1)

        if dist.is_initialized():
            dist.broadcast(init, 0)

        self.embeddings.data.copy_(init)
        self.z_avg.data.copy_(init)
        self.N.data.fill_(1.0)

        self._need_init = False

    # -----------------------------
    # INFERENCE
    # -----------------------------
    def dictionary_lookup(self, encodings):
        emb_norm = F.normalize(self.embeddings, dim=1)
        return F.embedding(encodings, emb_norm)
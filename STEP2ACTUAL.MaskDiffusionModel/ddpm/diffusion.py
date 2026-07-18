from skimage.measure import marching_cubes, mesh_surface_area
from scipy.ndimage import label
import json
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

import nibabel as nib
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(
            0, 1, 1), padding_mode="replicate")
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8,
        num_organs=9,
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding), padding_mode="replicate")

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning (diffusion timestep)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # delta_t conditioning (longitudinal time gap between ct0 and ct1),
        # sinusoidally embedded then projected to time_dim

        self.num_organs = num_organs

        self.delta_t_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # combine delta_t embedding with organ one-hot

        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + self.num_organs, time_dim),
            nn.SiLU(),
            nn.LayerNorm(time_dim),
            nn.Linear(time_dim, time_dim)
        )

        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)

        # If scale is 1 or no condition is passed, return standard logits
        if cond_scale == 1 or 'cond' not in kwargs or kwargs['cond'] is None:
            return logits

        # Unconditional pass: drop 100% of the conditions
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)

        # CFG Extrapolation
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        delta_t=None,
        organ=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        prob_focus_present=0.
    ):
        batch, device = x.shape[0], x.device

        drop_mask = torch.rand(batch, device=device) < null_cond_prob

        # if exists(cond):
        #    spatial_mask = drop_mask.view(batch, 1, 1, 1, 1)
        #    cond = torch.where(spatial_mask, torch.zeros_like(cond), cond)
        x = torch.cat([x, cond], dim=1)

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        # 3. Handle diffusion timestep, then delta_t + organ embedding
        t = self.time_mlp(time)

        if delta_t is not None and organ is not None:
            if delta_t.dim() == 1:
                delta_t = delta_t.view(batch)

            delta_t_emb = self.delta_t_mlp(delta_t)  # Shape: (B, time_emb_dim)

            # organ is expected as a one-hot float tensor (B, num_organs)
            cond_input = torch.cat([delta_t_emb, organ], dim=1)
            tab_emb = self.cond_mlp(cond_input)  # Shape: (B, time_emb_dim)

            # Reshape mask to (B, 1) for the embedding
            emb_mask = drop_mask.view(batch, 1)

            # Zero out the combined embedding for the dropped batch items
            tab_emb = torch.where(emb_mask, torch.zeros_like(tab_emb), tab_emb)

            # Add to the timestep embedding
            t = t + tab_emb

        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, delta_t=None, organ=None, cond_scale=1.):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(
                x, t, cond=cond, delta_t=delta_t, organ=organ, cond_scale=cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, delta_t=None, organ=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, delta_t=delta_t, organ=organ, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, delta_t=None, organ=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, delta_t=delta_t, organ=organ, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, heatmap, tumor_mask_0, organ_mask_0, delta_t=None, organ=None, cond_scale=1., batch_size=None):
        device = next(self.denoise_fn.parameters()).device

        # Match forward(): permute to (B, C, D, H, W), then concat tumor_mask_0 + organ_mask_0 + heatmap
        tumor_mask_0 = tumor_mask_0.permute(0, 1, -1, -3, -2).to(device)
        organ_mask_0 = organ_mask_0.permute(0, 1, -1, -3, -2).to(device)
        heatmap = heatmap.permute(0, 1, -1, -3, -2).to(device)
        cond = torch.cat([tumor_mask_0, organ_mask_0, heatmap], dim=1)

        if delta_t is not None:
            delta_t = delta_t.to(device)
        if organ is not None:
            organ = organ.to(device)

        if batch_size is None:
            batch_size = cond.shape[0]

        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames

        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size),
            cond=cond,
            delta_t=delta_t,
            organ=organ,
            cond_scale=cond_scale
        )

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, delta_t=None, organ=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(
            x_noisy, t, cond=cond, delta_t=delta_t, organ=organ, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, heatmap, tumor_mask_0, tumor_mask_1, organ_mask_0, delta_t, organ=None, null_cond_prob=0.0, *args, **kwargs):
        # 1. Permute all inputs for 3D processing (B, C, D, H, W)
        tumor_mask_0 = tumor_mask_0.permute(0, 1, -1, -3, -2)
        tumor_mask_1 = tumor_mask_1.permute(0, 1, -1, -3, -2)

        organ_mask_0 = organ_mask_0.permute(0, 1, -1, -3, -2)
        heatmap = heatmap.permute(0, 1, -1, -3, -2)

        target_mask = tumor_mask_1

        # 2. Concatenate spatial conditions along the channel dimension (dim=1)
        #    tumor_mask_0 anchors "where the tumor was", organ_mask_0 anchors
        #    organ geometry, heatmap (built from tumor_mask_1) tells the model
        #    where the target tumor should end up.
        cond = torch.cat([tumor_mask_0, organ_mask_0, heatmap], dim=1)

        # 3. Apply diffusion to the TARGET MASK (tumor_mask_1)
        b, device = target_mask.shape[0], target_mask.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        delta_t = delta_t.to(device).float()

        return self.p_losses(target_mask, t, cond=cond, delta_t=delta_t, organ=organ,
                              null_cond_prob=null_cond_prob, *args, **kwargs)

# trainer class

def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


# --- NEW IMPORTS REQUIRED FOR METRICS ---


MASK_COLUMNS = ['diameter_x_mm', 
                'diameter_y_mm', 
                'diameter_z_mm',
                'original_shape_Elongation',
                'original_shape_Flatness',
                'original_shape_LeastAxisLength',
                'original_shape_MajorAxisLength',
                'original_shape_Maximum2DDiameterColumn',
                'original_shape_Maximum2DDiameterRow',
                'original_shape_Maximum2DDiameterSlice',
                'original_shape_Maximum3DDiameter',
                'original_shape_MeshVolume',
                'original_shape_MinorAxisLength',
                'original_shape_Sphericity',
                'original_shape_SurfaceArea',
                'original_shape_SurfaceVolumeRatio',
                'original_shape_VoxelVolume']


def compute_voxel_volume(mask, spacing):
    """
    Computes:
        - original_shape_VoxelVolume (mm^3)
        - num_components

    Equivalent to PyRadiomics VoxelVolume for the entire mask.
    """
    if hasattr(mask, "numpy"):
        mask = mask.cpu().numpy()

    mask = np.squeeze(mask)
    spacing = np.array(spacing)

    bin_mask = mask > 0

    if not bin_mask.any():
        return {
            "original_shape_VoxelVolume": 0.0,
            "num_components": 0,
        }

    # Connected components
    structure = np.ones((3, 3, 3), dtype=bool)
    _, num_components = label(bin_mask, structure=structure)

    # Voxel volume
    voxel_volume_mm3 = np.prod(spacing)
    volume_mm3 = np.sum(bin_mask) * voxel_volume_mm3

    return {
        "original_shape_VoxelVolume": float(volume_mm3),
        "num_components": int(num_components),
    }



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
        voxel_spacing=(1.0, 1.0, 1.0),
        num_organs=9,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.cfg = cfg
        dl = dataset

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)


        self.opt = Adam([
            {'params': [p for n, p in diffusion_model.denoise_fn.named_parameters()
                        if 'cond_mlp' not in n and 'delta_t_mlp' not in n]},
            {'params': diffusion_model.denoise_fn.cond_mlp.parameters(), 'lr': train_lr * 5}, 
            {'params': diffusion_model.denoise_fn.delta_t_mlp.parameters(), 'lr': train_lr * 20}, # v1 = 5 and v2 = 20
        ], lr=cfg.model.train_lr)

        #self.opt = Adam(diffusion_model.denoise_fn.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(str(self.results_folder)+'/logs'):
            os.makedirs(str(self.results_folder)+'/logs')
        self.writer = SummaryWriter(str(self.results_folder)+'/logs')

        self.num_organs = num_organs

        self.reset_parameters()

        self.space_x, self.space_y, self.space_z = voxel_spacing

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'{milestone}.pt'))

    def load(self, milestone=-1, map_location=None, **kwargs):
        if milestone == -1:
            checkpoints = [
                int(p.stem)
                for p in Path(self.results_folder).glob("*.pt")
                if p.stem.isdigit()
            ]

            assert checkpoints, (
                "Need to have at least one numeric checkpoint to load "
                "from the latest checkpoint (milestone == -1)."
            )

            milestone = max(checkpoints)

        checkpoint_path = Path(self.results_folder) / f"{milestone}.pt"

        if map_location is not None:
            data = torch.load(checkpoint_path, map_location=map_location)
        else:
            data = torch.load(checkpoint_path)

        self.step = data["step"]
        self.model.load_state_dict(data["model"], **kwargs)
        self.ema_model.load_state_dict(data["ema"], **kwargs)
        self.scaler.load_state_dict(data["scaler"])

    def prepare_organ_one_hot(self, data, device):
        """
        One-hot encodes the organ class from the dataloader's `organ_id`
        field. Output shape: (Batch, num_organs)

        NOTE: this assumes the dataloader supplies an integer `organ_id`
        key (see organ_mapping in the data-loading code). If your data
        dict only has the string `organ` key, map it to `organ_id` in the
        dataset/collate step first — F.one_hot needs integer class indices,
        not strings.
        """
        organ_idx = torch.as_tensor(
            data["organ_id"], dtype=torch.long, device=device).view(-1)

        organ_one_hot = F.one_hot(
            organ_idx, num_classes=self.num_organs).float()

        return organ_one_hot

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        best_train_loss = 0.90
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)

                # Prepare input & conditioners
                tumor_mask_0 = data['tumor_mask_0'].cuda()
                tumor_mask_1 = data['tumor_mask_1'].cuda()

                organ_mask_0 = data['organ_mask_0'].cuda()
                organ_mask_1 = data["organ_mask_1"].cuda()

                heatmap = data["heatmap"].cuda()
                delta_t = data["delta_t"].cuda().float()

                device = heatmap.device

                organ = self.prepare_organ_one_hot(data, device=device)

                with autocast(enabled=self.amp):
                    loss = self.model(
                        heatmap=heatmap,
                        tumor_mask_0=tumor_mask_0,
                        tumor_mask_1=tumor_mask_1,           # Target for diffusion!
                        organ_mask_0=organ_mask_0,           # Spatial condition
                        delta_t=delta_t,                     # Continuous condition
                        organ=organ,                         # One-hot condition
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                        null_cond_prob=0.1
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                if (self.step % 10 == 0):
                    print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            lr = self.opt.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('Train_Loss', loss.item(), self.step)
            self.writer.add_scalar('Learning_rate', lr, self.step)

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    self.save('model_best')
                    print(f'New best model found at step {self.step}')

            if self.step > 0 and self.step % self.update_ema_every == 0:
                self.step_ema()

            # ==========================================
            # DEBUG: GENERATE AND SAVE NIFTI MASKS
            # ==========================================
            if self.step % 2000 == 0:
                print(
                    f"--> [Step {self.step}] Generating debug NIfTI reconstructions...")
                self.ema_model.eval()

                debug_folder = self.results_folder / 'debug_masks'
                debug_folder.mkdir(exist_ok=True)

                with torch.no_grad():
                    # 1. Permute to match forward()
                    heatmap_p = heatmap.permute(0, 1, -1, -3, -2)
                    tumor_mask_0_p = tumor_mask_0.permute(0, 1, -1, -3, -2)
                    organ_mask_0_p = organ_mask_0.permute(0, 1, -1, -3, -2)

                    cond = torch.cat(
                        [tumor_mask_0_p, organ_mask_0_p, heatmap_p], dim=1)

                    T_START = self.ema_model.num_timesteps

                    noisy_latent = torch.randn_like(tumor_mask_0_p)

                    recon_latent = noisy_latent
                    for i in tqdm(reversed(range(T_START))):
                        t_i = torch.full(
                            (recon_latent.shape[0],), i, device=recon_latent.device, dtype=torch.long)

                        recon_latent = self.ema_model.p_sample(
                            recon_latent, t_i, cond=cond, delta_t=delta_t, organ=organ, cond_scale=2.0)

                    recon = recon_latent.permute(0, 1, -2, -1, -3)

                    # Map from [-1, 1] back to [0, 1]
                    recon_normalized = (recon + 1.0) / 2.0

                    # Threshold back to binary
                    generated_masks = (recon_normalized < 0.5).float()

                    masks_np = generated_masks.cpu().numpy().astype(np.uint8)
                    raw_np = recon_normalized.cpu().numpy()

                    for b_idx in range(min(3, masks_np.shape[0])):
                        pred_3d = masks_np[b_idx, 0, :, :, :]
                        raw_3d = raw_np[b_idx, 0, :, :, :]

                        real_raw = tumor_mask_1[b_idx, 0, :, :, :].cpu().numpy()

                        if pred_3d.sum() > 0:
                            affine = np.array([
                                [self.space_x, 0, 0, 0],
                                [0, self.space_y, 0, 0],
                                [0, 0, self.space_z, 0],
                                [0, 0, 0, 1]
                            ])

                            nib.save(
                                nib.Nifti1Image(pred_3d, affine=affine),
                                str(debug_folder /
                                    f"step_{self.step}_sample_{b_idx}_RECON.nii.gz")
                            )
                            nib.save(
                                nib.Nifti1Image(raw_3d, affine=affine),
                                str(debug_folder /
                                    f"step_{self.step}_sample_{b_idx}_RAW_RECON.nii.gz")
                            )
                            nib.save(
                                nib.Nifti1Image(real_raw, affine=affine),
                                str(debug_folder /
                                    f"step_{self.step}_sample_{b_idx}_REAL_RAW.nii.gz")
                            )
                self.ema_model.train()

            log_fn(log)
            self.step += 1
        print('training completed')

class Tester(object):
    def __init__(
        self,
        diffusion_model,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema_model = copy.deepcopy(self.model)
        self.step = 0
        self.image_size = diffusion_model.image_size

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)

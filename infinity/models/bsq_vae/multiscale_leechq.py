"""
Spherical Leech Quantization
Proposed in https://arxiv.org/abs/2512.14697
Implemented with VQ with a fixed codebook
"""

import math
import random
from math import log2, ceil
from functools import partial, cache
from collections import namedtuple
from contextlib import nullcontext

import numpy as np
import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange, reduce, pack, unpack

from .dynamic_resolution import predefined_HW_Scales_dynamic

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

# entropy

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# cosine sim linear

class CosineSimLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale


def get_latent2scale_schedule(T: int, H: int, W: int, mode="original"):
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3", "half", "dense_f8"]
    predefined_HW_Scales = {
        # 256 * 256
        (32, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 9), (13, 13), (18, 18), (24, 24), (32, 32)],
        (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
        # 1024x1024
        (64, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), (12, 12), (16, 16), (21, 21), (27, 27), (36, 36), (48, 48), (64, 64)],

        (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
    }
    if mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode == "dense_f8":
        # predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = [(x, x) for x in range(1, 16+1)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
        predefined_HW_Scales[(128, 128)] = predefined_HW_Scales[(64, 64)] + [(80, 80), (96, 96), (112, 112), (128, 128)]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]
    elif mode == "half":
        predefined_HW_Scales[(32, 32)] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        predefined_HW_Scales[(64, 64)] = [(1,1),(2,2),(4,4),(6,6),(8,8),(12,12),(16,16)]

    predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # print("warning: the length of predefined_T_Scales is less than the length of patch_THW_shape_per_scale!")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w ) for (h, w), t in zip(patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale


class MultiScaleLeechQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        codebook_size,
        dim,
        soft_clamp_input_value = None,
        aux_loss = False, # intermediate auxiliary loss
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original", # ["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip = False,
        flip_prob = 0.5,
        flip_mode = "stochastic", # "stochastic", "deterministic"
        max_flip_lvl = 1,
        random_flip_1lvl = False, # random flip one level each time
        flip_lvl_idx = None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        random_short_schedule = False, # randomly use short schedule (schedule for images of 256x256)
        short_schedule_prob = 0.5,
        disable_flip_prob = 0.0, # disable random flip in this image
        uniform_short_schedule = False,
        leech_type = "full",
        **kwargs
    ):
        super().__init__()
        codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.layernorm = nn.Identity()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.disable_flip_prob = disable_flip_prob

        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0
        self.random_short_schedule = random_short_schedule
        self.short_schedule_prob = short_schedule_prob
        self.full2short = {7:7, 10:7, 13:7, 16:16, 20:16, 24:16}
        self.full2short_f8 = {20:20, 24:20, 28:20}
        self.uniform_short_schedule = uniform_short_schedule
        assert not (self.random_short_schedule and self.uniform_short_schedule)


        if leech_type == "full":
            codebook = "./cache/leech_lattices_normalized.npy"
        elif leech_type == "2":
            codebook = "./cache/leech_lattices_type2_normalized.npy"
        elif leech_type == "3":
            codebook = "./cache/leech_lattices_type3_normalized.npy"
        elif leech_type == "4":
            codebook = "./cache/leech_lattices_type4_normalized.npy"
        self.lfq = SphericalVectorQuantizer(
            n_embed=codebook_size,
            embed_dim=codebook_dim,
            l2_norm=True,
            predefined_codebook=codebook,
            grad="ste",
            requires_grad=False,
        )
        self.lfq.reset_parameters()

        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'
        
        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def flip_quant(self, x):
        if self.flip_mode == 'stochastic':
            flip_mask = torch.rand_like(x) < self.flip_prob
        else:
            raise NotImplementedError
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        scale_schedule=None,
        mask = None,
        return_all_codes = False,
        return_residual_norm_per_scale = False,
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()    
        
        if self.schedule_mode.startswith("same"):
            scale_num = int(self.schedule_mode[len("same"):])
            assert T == 1
            scale_schedule = [(1, H, W)] * scale_num
        else:
            scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
            scale_num = len(scale_schedule)
        
        if self.uniform_short_schedule:
            scale_num_short = self.full2short_f8[scale_num] if self.schedule_mode == "dense_f8" else self.full2short[scale_num]
            scale_num = random.randint(scale_num_short, scale_num)
            scale_schedule = scale_schedule[:scale_num]
        elif self.random_short_schedule and random.random() < self.short_schedule_prob:
            if self.schedule_mode == "dense_f8":
                scale_num = self.full2short_f8[scale_num]
            else:
                scale_num = self.full2short[scale_num]
            scale_schedule = scale_schedule[:scale_num]


        if self.has_projections:
            # x = self.project_in(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
            x = self.project_in(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w) 

        x = self.layernorm(x)

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_bit_indices = []
        var_inputs = []
        residual_norm_per_scale = []
        
        # go through the layers
        out_fact = init_out_fact = 1.0
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        if self.drop_when_test:
            drop_lvl_start = self.drop_lvl_idx
            drop_lvl_end = self.drop_lvl_idx + self.drop_lvl_num
        disable_flip = True if random.random() < self.disable_flip_prob else False # disable random flip in this image
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                out_fact = max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                if (pt, ph, pw) != (T, H, W):
                    interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    interpolate_residual = residual
                if return_residual_norm_per_scale:
                    residual_norm_per_scale.append((torch.abs(interpolate_residual) < 0.05 * self.lfq.codebook_scale).sum() / interpolate_residual.numel())
                # residual_list.append(torch.norm(residual.detach(), dim=1).mean())
                # interpolate_residual_list.append(torch.norm(interpolate_residual.detach(), dim=1).mean())
                if self.training and self.use_stochastic_depth and random.random() < self.drop_rate:
                    if (si == 0 and self.keep_first_quant) or (si == scale_num - 1 and self.keep_last_quant):
                        # interpolate_residual = rearrange(interpolate_residual, 'b c t h w -> b (t h w) c')
                        # interpolate_residual = F.normalize(interpolate_residual, dim=-1)
                        quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                        # quantized = rearrange(quantized, 'b (t h w) c -> b c t h w', t=pt, h=ph, w=pw)
                        if self.random_flip and si < self.max_flip_lvl and (not disable_flip):
                            quantized = self.flip_quant(quantized)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                        all_bit_indices.append(bit_indices)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue                     
                else:
                    # residual_norm = torch.norm(interpolate_residual.detach(), dim=1) # (b, t, h, w)
                    # print(si, residual_norm.min(), residual_norm.max(), residual_norm.mean())
                    # interpolate_residual = rearrange(interpolate_residual, 'b c t h w -> b (t h w) c')
                    # interpolate_residual = F.normalize(interpolate_residual, dim=-1)
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                    # quantized = rearrange(quantized, 'b (t h w) c -> b c t h w', t=pt, h=ph, w=pw)
                    if self.random_flip and si < self.max_flip_lvl and (not disable_flip):
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx and (not disable_flip):
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                    all_losses.append(loss)
                    all_bit_indices.append(bit_indices)
                # quantized_list.append(torch.norm(quantized.detach(), dim=1).mean())
                if (pt, ph, pw) != (T, H, W):
                    quantized = F.interpolate(quantized, size=(T, H, W), mode=self.z_interplote_up).contiguous()
                
                if self.remove_residual_detach:
                    residual = residual - quantized
                else:
                    residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                if si != scale_num - 1:
                    var_inputs.append(F.interpolate(quantized_out, size=scale_schedule[si+1], mode=self.z_interplote_down).contiguous())

                if self.use_decay_factor:
                    out_fact -= 0.1
        # print("residual_list:", residual_list)
        # print("interpolate_residual_list:", interpolate_residual_list)
        # print("quantized_list:", quantized_list)
        # import ipdb; ipdb.set_trace()
        # project out, if needed
        if self.has_projections:
            quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
            quantized_out = self.project_out(quantized_out)
            quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w)
            

        # image
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)

        # stack all losses and indices

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out, all_indices, all_bit_indices, residual_norm_per_scale, all_losses, var_inputs)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


class SphericalVectorQuantizer(Module):
    def __init__(
        self, n_embed, embed_dim, l2_norm=True, predefined_codebook=None, grad="ste", requires_grad=False
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.requires_grad = requires_grad

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.predefined_codebook = predefined_codebook

        assert grad in ["ste"]
        self.grad = grad

        self.channel_first = None

    def reset_parameters(self):
        if self.predefined_codebook is not None:
            print(f"Loading predefined codebook from {self.predefined_codebook}")
            weights = np.load(self.predefined_codebook)            
            assert weights.shape[1] == self.embed_dim
            if weights.shape[0] == self.n_embed:
                self.embedding.weight.data.copy_(torch.from_numpy(weights))
            elif weights.shape[0] < self.n_embed:
                raise ValueError(f"The number of vectors in the predefined codebook ({weights.shape[0]}) is less than n_embed ({self.n_embed})")
            else:
                print(f"Warning: The number of vectors in the predefined codebook ({weights.shape[0]}) is larger than n_embed ({self.n_embed}). Selecting the first {self.n_embed} vectors.")
                selected_indices = np.arange(self.n_embed)
                self.embedding.weight.data.copy_(torch.from_numpy(weights[selected_indices, :]))
        else:
            nn.init.trunc_normal_(self.embedding.weight, std=0.02)
            # raise ValueError(f"No predefined codebook provided")

    def forward(self, z):
        is_img_or_video = z.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        if should_transpose:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d') # x.shape [b, hwt, c]

        b, l, d = z.shape
        if self.l2_norm:
            z_flatten = z.reshape(-1, self.embed_dim)
            if self.predefined_codebook is None:
                embedding_weight = F.normalize(self.embedding.weight, dim=-1)
                d = -z_flatten @ embedding_weight.t()
            else:
                d = -z_flatten @ self.embedding.weight.t()
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            d = (
                torch.sum(z_flatten**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * z_flatten @ self.embedding.weight.t()
            )

        min_encoding_indices = torch.argmin(d.detach(), dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        z_q = z + (z_q - z).detach()

        if not self.training:
            used_codes = torch.unique(min_encoding_indices, return_counts=False)
        else:
            used_codes = None
        cb_usage = torch.bincount(
            min_encoding_indices.long(), minlength=self.n_embed
        ).float()
        cb_entropy = self.get_entropy(cb_usage)
        loss = torch.zeros(1, device=z.device, dtype=z.dtype)

        if should_transpose:
            z_q = unpack_one(z_q, ps, 'b * d')
            bit_indices = (z_q * math.sqrt(32)).long() + 4
            z_q = rearrange(z_q, 'b ... d -> b d ...')

        ret = (z_q, min_encoding_indices, bit_indices, loss)
        return ret

    def get_entropy(self, count, eps=1e-4):
        probs = (count + eps) / (count + eps).sum()
        return -torch.sum(probs * torch.log(probs))

    def indices_to_codes(self, indices, label_type='int_label'):
        assert label_type in ['int_label', 'dit_label']
        if label_type == 'int_label':
            b, _, h, w = indices.shape

            z_q = self.embedding(indices.flatten())
            if self.l2_norm:
                z_q = F.normalize(z_q, dim=-1)
            codes = rearrange(z_q, "(b h w) d -> b d h w", h=h, w=w)
            return codes.unsqueeze(2)
        else:
            indice_shifted = (indices - 4)
            indice_norm = indice_shifted / torch.norm(indice_shifted.float(), dim=-1, keepdim=True)
            d = ((indice_norm - self.embedding.weight.data.unsqueeze(1).unsqueeze(1)) ** 2).sum(-1)
            min_encoding_indices = torch.argmin(d, dim=1, keepdim=True)
            return self.indices_to_codes(min_encoding_indices, label_type='int_label')

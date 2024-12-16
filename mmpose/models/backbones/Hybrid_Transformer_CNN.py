import torch.utils.checkpoint as checkpoint
from torch import Tensor, Size
from typing import Union, List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init,build_conv_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple
from functools import partial
from mmpose.utils import get_root_logger
from mmpose.models.builder import BACKBONES
from mmpose.models.utils import PatchEmbed_new, nchw_to_nlc, nlc_to_nchw, pvt_convert
from mmpose.models.backbones.utils import load_checkpoint

from mmpose.models.builder import BACKBONES

# from mmcv.utils import get_logger

# from utils.ckpt_load import load_checkpoint_swin


# def get_root_logger(log_file=None, log_level=logging.INFO):
#     """Use `get_logger` method in mmcv to get the root logger.
#
#     The logger will be initialized if it has not been initialized. By default a
#     StreamHandler will be added. If `log_file` is specified, a FileHandler will
#     also be added. The name of the root logger is the top-level package name,
#     e.g., "mmpose".
#
#     Args:
#         log_file (str | None): The log filename. If specified, a FileHandler
#             will be added to the root logger.
#         log_level (int): The root logger level. Note that only the process of
#             rank 0 is affected, while other processes will set the level to
#             "Error" and be silent most of the time.
#
#     Returns:
#         logging.Logger: The root logger.
#     """
# return get_logger(__name__.split('.')[0], log_file, log_level)


_shape_t = Union[int, List[int], Size]


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=None):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNormFP32(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormFP32, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)


class LinearFP32(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearFP32, self).__init__(in_features, out_features, bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float(),
                        self.bias.float() if self.bias is not None else None)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=None, mlpfp32=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        if self.mlpfp32:
            x = self.fc2.float()(x.type(torch.float32))
            x = self.drop.float()(x)
            # print(f"======>[MLP FP32]")
        else:
            x = self.fc2(x)
            x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=None, mlpfp32=False, proj_ln=False):
        super().__init__()
        self.mlp = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
                       act_layer=act_layer, drop=drop, norm_layer=norm_layer, mlpfp32=mlpfp32)
        self.conv_proj = nn.Conv2d(in_features,
                                   in_features,
                                   kernel_size=3,
                                   padding=1,
                                   stride=1,
                                   bias=False,
                                   groups=in_features)
        self.proj_ln = LayerNorm2D(in_features, LayerNormFP32) if proj_ln else None

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        x = self.conv_proj(x)
        if self.proj_ln:
            x = self.proj_ln(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.reshape(B, L, C)
        x = self.mlp(x, H, W)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlpfp32=False, pretrain_window_size=-1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.mlpfp32 = mlpfp32
        self.attn_type = attn_type
        self.rpe_output_type = rpe_output_type
        self.relative_coords_table_type = relative_coords_table_type

        if self.attn_type == 'cosine_mh':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        elif self.attn_type == 'normal':
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
        else:
            raise NotImplementedError()
        if self.relative_coords_table_type != "none":
            # mlp to generate table of relative position bias
            self.rpe_mlp = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                         nn.ReLU(inplace=True),
                                         LinearFP32(rpe_hidden_dim, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if relative_coords_table_type == 'linear':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            elif relative_coords_table_type == 'linear_bylayer':
                print(f"norm8_log_bylayer: [{self.window_size}] ==> [{pretrain_window_size}]")
                relative_coords_table[:, :, :, 0] /= (pretrain_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrain_window_size - 1)
            elif relative_coords_table_type == 'norm8_log':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_192to640':
                if self.window_size[0] == 40:
                    relative_coords_table[:, :, :, 0] /= (11)
                    relative_coords_table[:, :, :, 1] /= (11)
                elif self.window_size[0] == 20:
                    relative_coords_table[:, :, :, 0] /= (5)
                    relative_coords_table[:, :, :, 1] /= (5)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            # check
            elif relative_coords_table_type == 'norm8_log_256to640':
                if self.window_size[0] == 40:
                    relative_coords_table[:, :, :, 0] /= (15)
                    relative_coords_table[:, :, :, 1] /= (15)
                elif self.window_size[0] == 20:
                    relative_coords_table[:, :, :, 0] /= (7)
                    relative_coords_table[:, :, :, 1] /= (7)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_bylayer':
                print(f"norm8_log_bylayer: [{self.window_size}] ==> [{pretrain_window_size}]")
                relative_coords_table[:, :, :, 0] /= (pretrain_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrain_window_size - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            else:
                raise NotImplementedError
            self.register_buffer("relative_coords_table", relative_coords_table)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.attn_type == 'cosine_mh':
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            logit_scale = torch.clamp(self.logit_scale,
                                      max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
            attn = (q @ k.transpose(-2, -1)) * logit_scale.float()
        elif self.attn_type == 'normal':
            q = q * self.scale
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            raise NotImplementedError()

        if self.relative_coords_table_type != "none":
            # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
            relative_position_bias_table = self.rpe_mlp(self.relative_coords_table).view(-1, self.num_heads)
        else:
            relative_position_bias_table = self.relative_position_bias_table
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.rpe_output_type == 'normal':
            pass
        elif self.rpe_output_type == 'sigmoid':
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        else:
            raise NotImplementedError

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.mlpfp32:
            x = self.proj.float()(x.type(torch.float32))
            x = self.proj_drop.float()(x)
            # print(f"======>[ATTN FP32]")
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlockPost(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_mlp_norm=False, endnorm=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlp_type='normal', mlpfp32=False,
                 pretrain_window_size=-1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type, rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim, attn_type=attn_type, mlpfp32=mlpfp32,
            pretrain_window_size=pretrain_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if mlp_type == 'normal':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv_ln':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32, proj_ln=True)

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x

        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.norm1.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.mlp(x, H, W)
        if self.mlpfp32:
            x = self.norm2.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm2(x)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x


class SwinTransformerBlockPre(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_mlp_norm=False, endnorm=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 init_values=None, relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlp_type='normal', mlpfp32=False,
                 pretrain_window_size=-1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type, rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim, attn_type=attn_type, mlpfp32=mlpfp32,
            pretrain_window_size=pretrain_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if mlp_type == 'normal':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv_ln':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32, proj_ln=True)

        if init_values is not None and init_values >= 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.gamma_1 * x
            x = x.type(orig_type)
        else:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.norm2(x)
        if self.mlpfp32:
            x = self.gamma_2 * self.mlp(x, H, W)
            x = x.type(orig_type)
        else:
            x = self.gamma_2 * self.mlp(x, H, W)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # print('x.size',x.size())
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)
        # print('x.size', x.size())
        return x


class PatchReduction1C(nn.Module):
    r""" Patch Reduction Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x


class ConvPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        if self.postnorm:
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_shift (bool): Whether to use shifted window. Default: True.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 checkpoint_blocks=255,
                 init_values=None,
                 endnorm_interval=-1,
                 use_mlp_norm=False,
                 use_shift=True,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_output_type='normal',
                 attn_type='normal',
                 mlp_type='normal',
                 mlpfp32_blocks=[-1],
                 postnorm=True,
                 pretrain_window_size=-1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
        self.init_values = init_values if init_values is not None else 0.0
        self.endnorm_interval = endnorm_interval
        self.mlpfp32_blocks = mlpfp32_blocks
        self.postnorm = postnorm

        # build blocks
        if self.postnorm:
            self.blocks = nn.ModuleList([
                SwinTransformerBlockPost(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_mlp_norm=use_mlp_norm,
                    endnorm=True if ((i + 1) % endnorm_interval == 0) and (
                            endnorm_interval > 0) else False,
                    relative_coords_table_type=relative_coords_table_type,
                    rpe_hidden_dim=rpe_hidden_dim,
                    rpe_output_type=rpe_output_type,
                    attn_type=attn_type,
                    mlp_type=mlp_type,
                    mlpfp32=True if i in mlpfp32_blocks else False,
                    pretrain_window_size=pretrain_window_size)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                SwinTransformerBlockPre(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_mlp_norm=use_mlp_norm,
                    endnorm=True if ((i + 1) % endnorm_interval == 0) and (
                            endnorm_interval > 0) else False,
                    relative_coords_table_type=relative_coords_table_type,
                    rpe_hidden_dim=rpe_hidden_dim,
                    rpe_output_type=rpe_output_type,
                    attn_type=attn_type,
                    mlp_type=mlp_type,
                    mlpfp32=True if i in mlpfp32_blocks else False,
                    pretrain_window_size=pretrain_window_size)
                for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, postnorm=postnorm)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        bs = x.size()[0]
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for idx, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        # if self.downsample is not None:
        #     x_down = self.downsample(x, H, W)
        #     if isinstance(self.downsample, PatchReduction1C):
        #         return x, H, W, x_down, H, W
        #     else:
        #         Wh, Ww = (H + 1) // 2, (W + 1) // 2
        #         print('downsample is not None')
        #         print(x.size(), H, W, x_down.size(), Wh, Ww)
        #         return x, H, W, x_down, Wh, Ww
        # else:
        #     print('downsample is  None')
        #     return x, H, W, x, H, W
        return x.view(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def _init_block_norm_weights(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, self.init_values)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, self.init_values)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        # print('x1',x.size())#x torch.Size([2, 3, 256, 256])

        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        # print('x2', x.size())#x torch.Size([2, 128, 64, 64])
        return x


class ResNetDLNPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False),
                                   LayerNorm2D(64, norm_layer),
                                   nn.GELU(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                                   LayerNorm2D(64, norm_layer),
                                   nn.GELU(),
                                   nn.Conv2d(64, embed_dim, 3, stride=1, padding=1, bias=False))
        self.norm = LayerNorm2D(embed_dim, norm_layer if norm_layer is not None else LayerNormFP32)  # use ln always
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.maxpool(x)
        # x = x.flatten(2).transpose(1, 2)
        return x


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv4 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = self.conv4(x)

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            pass
            # out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class HTC(BaseModule):
    """
    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 64.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        patch_sizes (Sequence[int]): The patch_size of each patch embedding.
            Default: [4, 2, 2, 2].
        strides (Sequence[int]): The stride of each patch embedding.
            Default: [4, 2, 2, 2].
        paddings (Sequence[int]): The padding of each patch embedding.
            Default: [0, 0, 0, 0].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer encode layer.
            Default: [8, 8, 4, 4].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: True.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 window_size=[16,16,16,8],
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 swin_num_heads=[2, 4, 10, 16],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 paddings=[0, 0, 0, 0],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=[8, 8, 4, 4],
                 chratio = [1,1,1,1],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 norm_after_stage=False,
                 use_shift=[True, True, False, False],
                 use_conv_ffn=False,
                 act_cfg=dict(type='GELU'),
                 norm_layer=partial(LayerNormFP32, eps=1e-6),
                 norm_cfg=dict(type='BN'),#dict(type='LN', eps=1e-6),
                 pretrained=None,
                 postnorm=True,
                 convert_weights=False,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_interpolation='geo',
                 rpe_output_type='sigmoid',
                 endnorm_interval=-1,
                 attn_type='cosine_mh',
                 mlp_type = 'normal',
                 checkpoint_blocks=[255, 255, 255, 255],
                 mlpfp32_layer_blocks=[[-1], [-1], [-1], [-1]],
                 pretrain_window_size=[-1, -1, -1, -1],
                 init_values=1e-5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.convert_weights = convert_weights
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        self.embed_dims = embed_dims
        self.relative_coords_table_type = relative_coords_table_type
        self.window_size = window_size
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.swin_num_heads = swin_num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.chratio = chratio
        self.rpe_hidden_dim = rpe_hidden_dim
        self.rpe_output_type = rpe_output_type
        self.attn_type = attn_type
        self.mlp_type = mlp_type
        self.postnorm = postnorm
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained
        if isinstance(use_checkpoint, list):
            pass
        elif isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint] * len(num_layers)

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        '''self.preconv = Bottleneck(
            in_channels=in_channels,
            out_channels=embed_dims,
        )'''
        self.conv_lay = ModuleList()
        for i in range(num_stages):
            embed_dims_ic = embed_dims * (num_heads[i])
            self.conv_lay.extend([
                BasicBlock(
                    in_channels=embed_dims_ic,
                    out_channels=embed_dims_ic,
                )
            ])
            embed_dims_ic = embed_dims_ic
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * (num_heads[i])
            patch_embed = PatchEmbed_new(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=True,
                norm_cfg=norm_cfg)
            layer = BasicLayer(
                dim=embed_dims_i,
                depth=num_layer,
                num_heads=self.swin_num_heads[i],# 2,4,10,16 num_heads[i]*2 num_heads[i]
                window_size=window_size[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(num_layers[:i]):sum(num_layers[:i + 1])],
                norm_layer=norm_layer,
                downsample=None,#cur_downsample_layer
                use_checkpoint=use_checkpoint[i],
                checkpoint_blocks=checkpoint_blocks[i],
                init_values=init_values,
                endnorm_interval=endnorm_interval,
                use_mlp_norm=False,#True if i in use_mlp_norm_layers else False,
                use_shift=use_shift[i],
                relative_coords_table_type=self.relative_coords_table_type,
                rpe_hidden_dim=self.rpe_hidden_dim,
                rpe_output_type=self.rpe_output_type,
                attn_type=self.attn_type,
                mlp_type=self.mlp_type,
                mlpfp32_blocks=mlpfp32_layer_blocks[i],
                postnorm=self.postnorm,
                pretrain_window_size=pretrain_window_size[i]
            )

            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            if norm_after_stage:
                norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            else:
                norm = nn.Identity()
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        constant_init(m.bias, 0)


    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = layer[1](x, Wh, Ww)
            x = layer[2](x)

            x = self.conv_lay[i](x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return outs


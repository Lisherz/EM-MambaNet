from functools import partial
from typing import Optional, Callable, Any
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class CrossMambaFusionBlock(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


class ConcatMambaFusionBlock(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = ConMB_SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x = self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from ..builder import NECKS

def N2WH(x, H, W):
    # x: B N C -> B C H W
    B, N, C = x.shape
    x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x

def WH2N(x):
    # x: B C H W -> B N C
    B, C, H, W = x.shape
    x = x.reshape(B, C, H*W).permute(0, 2, 1).contiguous()
    return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., ds_ratio=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        #
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        #
        self.pool1 = nn.AdaptiveAvgPool2d(7)
        self.p_norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H*W
        # B N C
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3).contiguous()
        #
        x_ = N2WH(x, H, W) 
        r1 = self.pool1(x_).reshape(B, C, -1).permute(0,2,1)
        r = r1
        #
        kv = self.kv(self.p_norm(r)).reshape(B, r.shape[1], 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]
        # -> B H N C
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # -> B h H*W C/h -> B C H W
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MSSAModule(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ds_ratio=4):
        super(MSSAModule, self).__init__()
        #
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.ms_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ds_ratio=ds_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, 
                       drop=drop)

    def forward(self, x):
        # feat : B C H W
        B, C, H, W = x.shape
        x = WH2N(x)
        x = x + self.drop_path(self.ms_attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # B N C -> B C H W
        return N2WH(x, H, W)

@NECKS.register_module()
class MSSAAdp(nn.Module):
    r"""Feature Pyramid Network.
    TODO: add doc
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(MSSAAdp, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels) # 4
        self.num_outs = num_outs # 5
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins # 4
            assert num_outs >= self.num_ins - start_level # 4 - 0
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        # change fpn_convs into TR structure
        self.fpn_convs = nn.ModuleList()
        # 0 ~ 4
        ds_ratios = [16, 8, 4, 2]
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            fpn_conv = MSSAModule(
                dim=out_channels, 
                num_heads=4, 
                mlp_ratio=2., 
                qkv_bias=False, 
                qk_scale=None, 
                drop=0., attn_drop=0., 
                drop_path=0.,
                ds_ratio=ds_ratios[i])

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet) # 5 - 4 + 0
        extra_levels = num_outs - self.backbone_end_level + self.start_level

        
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # 4
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if i == used_backbone_levels - 1:
                laterals[i] = self.fpn_convs[i](laterals[i])
            #
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])
            
        outs = laterals

        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
         
        return tuple(outs)


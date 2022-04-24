from functools import partial

import torch
import torch.nn as nn
import math
from einops import reduce, rearrange
from timm.models.registry import register_model
# from timm.models.vision_transformer import VisionTransformer, _cfg
from vit_models.Model.model_vision_transformer import VisionTransformer, _cfg
import torch.nn.functional as F

__all__ = [
    "tiny_patch16_224_hierarchical", "small_patch16_224_hierarchical", "base_patch16_224_hierarchical"
]


class TransformerHead(nn.Module):
    expansion = 1

    def __init__(self, token_dim, num_patches=196, num_classes=1000, stride=1):
        super(TransformerHead, self).__init__()

        self.token_dim = token_dim
        self.num_patches = num_patches
        self.num_classes = num_classes

        # To process patches
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.token_dim != self.expansion * self.token_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.token_dim, self.expansion * self.token_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.token_dim)
            )

        self.token_fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x):
        """
            x : (B, num_patches + 1, D) -> (B, C=num_classes)
        """
        cls_token, patch_tokens = x[:, 0], x[:, 1:]
        size = int(math.sqrt(x.shape[1]))  # 原patch的尺寸为[1,196]，为了卷积提取结构信息，创建为14*14的形式

        patch_tokens = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=size, w=size)  # B, D, H, W
        features = F.relu(self.bn(self.conv(patch_tokens)))  # H
        features = self.bn(self.conv(features))  # H
        features += self.shortcut(patch_tokens)  # H
        features = F.relu(features)  # H
        patch_tokens = F.avg_pool2d(features, 14).view(-1, self.token_dim)  # 14:kernel size, 前14列相加求平均，得[1,14]向量
        cls_token = self.token_fc(cls_token)

        out = patch_tokens + cls_token  # merge输出至classifier

        return out


class VisionTransformer_hierarchical(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Transformer heads
        self.transformerheads = nn.Sequential(*[
            TransformerHead(self.embed_dim)
            for i in range(11)])

    def forward_features(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # (B, -1, -1)中-1表示保持当前维度
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)  # 为了“保持期望不变”而进行了rescale(缩放），缩放比例是1/(1-p)，p是的丢弃比率。

        # Store transformer outputs
        transformerheads_outputs = []
        layer_attention = []
        for idx, blk in enumerate(self.blocks):
            x, a_ma = blk(x)  # transformer encoder的输出
            layer_attention.append(a_ma)
            # 此部分为非最后一个head的微调部分（本算法一共12个head，第0到第10个head需要经过trm模块才能送入classifier）
            if idx <= 10:
                out = self.norm(x)  # Final Local Norm
                out = self.transformerheads[idx](out)
                transformerheads_outputs.append(out)

        x = self.norm(x)
        return x, transformerheads_outputs, layer_attention

    def forward(self, x):
        x, transformerheads_outputs, patch_a_ma = self.forward_features(x)
        output = []
        for y in transformerheads_outputs:
            output.append(self.head(y))
        output.append(self.head(x[:, 0]))
        return output, patch_a_ma


@register_model
def tiny_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    """
            Args:
                patch_size (int, tuple): patch size
                embed_dim (int): embedding dimension
                depth (int): depth of transformer,block的数量
                num_heads (int): number of attention heads
                mlp_ratio (int): ratio of mlp hidden dim to embedding dim
                qkv_bias (bool): enable bias for qkv if True
                norm_layer: (nn.Module): normalization layer
            """
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_tiny_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def small_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_small_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def base_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_base_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model

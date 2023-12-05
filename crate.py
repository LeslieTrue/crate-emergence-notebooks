import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False, return_key = False):
        if return_key:
            return self.qkv(x)
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        if return_attention:
            return attn
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, dim, dropout = dropout, step_size=ista))
            ]))

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x

            x = ff(grad_x)
        return x

class CRATE(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., ista=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.depth = depth
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista)

        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.head = None
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        # self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature_last = x
        return self.mlp_head(x)

    def get_last_key(self, img, depth = 11):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        for i, (attn, ff) in enumerate(self.transformer.layers):
            if i < depth:
                grad_x = attn(x) + x
                x = ff(grad_x)
            else:
                key = attn(x, return_key = True)
                # print(key.shape)
                return key
        
    def get_last_selfattention(self, img, layer = 5):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        for i, (attn, ff) in enumerate(self.transformer.layers):
            if i < layer:
                grad_x = attn(x) + x
                x = ff(grad_x)
            else:
                attn_map = attn(x, return_attention=True)
                # print(attn_map.shape)
                return attn_map
        

def CRATE_base_demo():
    return CRATE(image_size=224,
            patch_size=8,
            num_classes=21842,
            dim=768,
            depth=12,
            heads=6,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=768//6)

def CRATE_small_21k():
    return CRATE(image_size=224,
                    patch_size=8,
                    num_classes=21842,
                    dim=384,
                    depth=12,
                    heads=6,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=384//6)

def CRATE_base_21k():
    return CRATE(image_size=224,
                    patch_size=8,
                    num_classes=21842,
                    dim=768,
                    depth=12,
                    heads=12,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=768//12)

def CRATE_base_768():
    return CRATE(image_size=224,
                    patch_size=8,
                    num_classes=768,
                    dim=768,
                    depth=12,
                    heads=12,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=768//12)
import math
from typing import Union, List, Tuple
import types
class CRATEFeat(nn.Module):
    def __init__(self,  feat_dim, pretrained_path = None, depth = 11, crate_arch = 'base', patch_size = 8, device = 'cpu'):
        super().__init__()
        if crate_arch == 'small':
            self.model = CRATE_small_21k()
        elif crate_arch == 'base':
            self.model = CRATE_base_768()
        elif crate_arch == 'demo':
            self.model = CRATE_base_demo()
            self.model.mlp_head = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768,768)
            )
            self.model.head = nn.Linear(768, 21842)
            
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.depth = depth
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict['model'], strict=False)
            print('Loading weight from {}'.format(pretrained_path))
        self.model = self.patch_vit_resolution(self.model, stride = 8)
        self.model.to(device)
        
    def patch_vit_resolution(self, model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_size
        stride = (stride, stride)
        model.interpolate_pos_encoding = types.MethodType(CRATEFeat._fix_pos_enc(patch_size, stride), model)
        return model
        
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embedding.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embedding
            class_pos_embed = self.pos_embedding[:, 0]
            patch_pos_embed = self.pos_embedding[:, 1:]
            dim = self.dim
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding
    def forward_attn(self, img, layer = 11):
        with torch.no_grad():
            
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            pos_em = self.model.interpolate_pos_encoding(img, w, h)
            self.model.pos_embedding = nn.Parameter(self.model.interpolate_pos_encoding(img, w, h))
            attentions = self.model.get_last_selfattention(img, layer = layer)
            return attentions
        
    def forward(self, img):
        with torch.no_grad():
            
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            pos_em = self.model.interpolate_pos_encoding(img, w, h)
            self.model.pos_embedding = nn.Parameter(self.model.interpolate_pos_encoding(img, w, h))
            attentions = self.model.get_last_selfattention(img, layer = self.depth)
            bs, nb_head, nb_token = attentions.shape[0], attentions.shape[1], attentions.shape[2]
            qkv = self.model.get_last_key(img, depth = self.depth)
            qkv = qkv[None, :, :, :]
            return qkv[:, :, 1:, :]
            
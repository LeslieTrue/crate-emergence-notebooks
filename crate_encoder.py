import torch
from torch import nn
from functools import partial

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.vision_transformer import PatchEmbed


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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
    def __init__(self, dim, hidden_dim, dropout = 0., step_size=0.1, lambd=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = lambd
        print("lambd:", self.lambd)

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

class Block_CRATE(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0., ista=0.1, lambd=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim = dim
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, dim, dropout = dropout, step_size=ista, lambd=lambd))
        ]))

    def forward(self, x, return_attention=False, return_key=False):
        depth = 0
        for attn, ff in self.layers:
            if return_attention:
                return attn(x, return_attention=True)
            if return_key:
                return attn(x, return_key=True)
            grad_x = attn(x) + x
            x = ff(grad_x)
        return x
import math

class CRATE(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, norm_layer=nn.LayerNorm, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., ista=0.1, lambd = 0.5):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embed = PatchEmbed(224, patch_height, 3, dim, strict_img_size=False)
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim), requires_grad=False) 

        self.blocks = nn.ModuleList([
            Block_CRATE(dim=dim, heads=heads, dim_head=dim//heads, lambd = lambd)
            for i in range(depth)])
        self.pool = pool
        self.to_latent = nn.Identity()

        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.decoder_pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.decoder_pos_embed[:, 0]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward(self, img):
        x = self.patch_embed(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed[:, :(n + 1)]
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        # feature_pre = x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # feature_last = x
        x = self.norm(x)
        x = self.head(x)
        return x
    
    def get_last_key_enc(self, img, layer = 11):
        B, nc, w, h = img.shape
        x = self.patch_embed(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            if i < layer:
                x = blk(x)
            else:
                key = blk(x, return_key = True)
                return key


def crate_tiny(**kwargs):
    num_classes = kwargs["num_classes"] if "num_classes" in kwargs else 1000
    return CRATE(image_size=224,
                    patch_size=16,
                    num_classes=num_classes,
                    dim=384,
                    depth=12,
                    heads=6,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=384//6, **kwargs)

def crate_small(**kwargs):
    num_classes = kwargs["num_classes"] if "num_classes" in kwargs else 1000
    return CRATE(image_size=224,
                    patch_size=16,
                    num_classes=num_classes,
                    dim=576,
                    depth=12,
                    heads=12,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=576//12, **kwargs)

def crate_base(**kwargs):
    num_classes = kwargs["num_classes"] if "num_classes" in kwargs else 1000
    return CRATE(image_size=224,
                patch_size=16,
                num_classes=num_classes,
                dim=768,
                depth=12,
                heads=12,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=768//12, **kwargs)

def crate_large(**kwargs):
    num_classes = kwargs["num_classes"] if "num_classes" in kwargs else 1000
    return CRATE(image_size=224,
                patch_size=16,
                num_classes=num_classes,
                dim=1024,
                depth=24,
                heads=16,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=1024//16, **kwargs)
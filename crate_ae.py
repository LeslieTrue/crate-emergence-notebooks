from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block as ViTBlock

from crate_decoder import Block_CRATE as CRATEDecoderBlock
from crate_encoder import Block_CRATE as CRATEEncoderBlock
from pos_embed import get_2d_sincos_pos_embed
from einops import rearrange, repeat


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(
            self, encoder_block, decoder_block, img_size=224, patch_size=16, in_chans=3,
            embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
    ):
        super().__init__()
        self.heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.dim = embed_dim

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        try:
            self.blocks = nn.ModuleList(
                [  # activates for CRATE blocks
                    encoder_block(dim=embed_dim, heads=num_heads, dim_head=embed_dim // num_heads)
                    for i in range(depth)]
            )
        except TypeError:
            self.blocks = nn.ModuleList(
                [  # activates for ViT blocks
                    encoder_block(dim=embed_dim, num_heads=num_heads)
                    for i in range(depth)]
            )

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        try:
            self.decoder_blocks = nn.ModuleList(
                [
                    decoder_block(
                        dim=decoder_embed_dim, heads=decoder_num_heads, dim_head=decoder_embed_dim // decoder_num_heads
                    )
                    for i in range(decoder_depth)]
            )
        except TypeError:
            self.decoder_blocks = nn.ModuleList(
                [
                    decoder_block(dim=decoder_embed_dim, num_heads=decoder_num_heads)
                    for i in range(decoder_depth)]
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.cls_token)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # append cls token back
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed (after putting mask tokens and input tokens together)
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, latent, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, latent, pred, mask)
        return loss, pred, mask


    def get_last_selfattention(self, img, layer = 12):
        x = self.patch_embed(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.decoder_pos_embed[:, :(n + 1)]

        # apply Transformer blocks
        # print(self.blocks)
        for i, crate_layer in enumerate(self.blocks):
            if i < layer:
              x = crate_layer(x)
            else:
              for _, (attn, ff) in enumerate(crate_layer.layers):
                attn_map = attn(x, return_attention=True)
                return attn_map

        #     print(layer)
        # for i, (attn, ff) in enumerate(self.blocks):
        #     if i < layer:
        #         grad_x = attn(x) + x
        #         x = ff(grad_x)
        #     else:
        #         attn_map = attn(x, return_attention=True)
        #         # print(attn_map.shape)
        #         return attn_map

def mae_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        ViTBlock, ViTBlock,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        ViTBlock, ViTBlock,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_huge(**kwargs):
    model = MaskedAutoencoderViT(
        ViTBlock, ViTBlock,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=1280, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_crate_tiny(**kwargs):
    model = MaskedAutoencoderViT(
        CRATEEncoderBlock, CRATEDecoderBlock,
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_crate_small(**kwargs):
    model = MaskedAutoencoderViT(
        CRATEEncoderBlock, CRATEDecoderBlock,
        patch_size=16, embed_dim=576, depth=12, num_heads=12,
        decoder_embed_dim=576, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_crate_base(**kwargs):
    model = MaskedAutoencoderViT(
        CRATEEncoderBlock, CRATEDecoderBlock,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_crate_large(**kwargs):
    model = MaskedAutoencoderViT(
        CRATEEncoderBlock, CRATEDecoderBlock,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=24, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

import math
from typing import Union, List, Tuple
import types
class CRATEFeat(nn.Module):
    def __init__(self,  feat_dim=768, pretrained_path = None, depth = 12, crate_arch = 'base', patch_size = 16, device = 'cpu'):
        super().__init__()
        if crate_arch == 'base':
            self.model = mae_crate_base()
        else:
            raise RuntimeError
            
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.depth = depth
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict['model'], strict=True)
            print('Loading weight from {}'.format(pretrained_path))
        # self.model = self.patch_vit_resolution(self.model, stride = 8)
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
            N = self.decoder_pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.decoder_pos_embed
            class_pos_embed = self.decoder_pos_embed[:, 0]
            patch_pos_embed = self.decoder_pos_embed[:, 1:]
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
    def forward_attn(self, img, layer = 12):
        with torch.no_grad():
            
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            # pos_em = self.model.interpolate_pos_encoding(img, w, h)
            # self.model.decoder_pos_embed = nn.Parameter(self.model.interpolate_pos_encoding(img, w, h))
            attentions = self.model.get_last_selfattention(img, layer = layer)
            return attentions

     
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.diffdecoder import DiffDecoder
from models.crossattention import CrossAttentionBlock, AttentionBlock

from diffusers.utils.torch_utils import randn_tensor
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from models.samplers import euler_maruyama_sampler


def _prepare_4d_attention_mask_for_sdpa(pre_len, seq_len, patch_size):
    dtype = torch.bfloat16
    attn_mask_converter = AttentionMaskConverter(is_causal=False, sliding_window=None)
    expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask_2d=torch.ones((1, pre_len + seq_len)),
            query_length=pre_len + seq_len,
            dtype=dtype,
        )
    expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
            expanded_4d_mask, min_dtype=torch.finfo(dtype).min
        )
    batch_size, _, query_length, _ = expanded_4d_mask.shape
    for batch_idx in range(batch_size):
        stride = [pre_len]+[seq_len//(patch_size**2) for _ in range(patch_size**2)]
        for n in range(1, len(stride)+1):
            x = sum(stride[:n])
            expanded_4d_mask[batch_idx, :, :x, x:] = torch.finfo(dtype).min
    return expanded_4d_mask.cuda()


class Transdiff(nn.Module):
    def __init__(self, img_size=256, patch_size=2,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.patch_size = patch_size

        self.seq_len = 256
        self.grad_checkpointing = grad_checkpointing
        self.token_embed_dim = vae_embed_dim * patch_size ** 2

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.mask_token = nn.Parameter(torch.zeros(1, 64, decoder_embed_dim))
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            AttentionBlock(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth * 2)])
        self.encoder_norm = norm_layer(encoder_embed_dim)
        self.attention_mask = _prepare_4d_attention_mask_for_sdpa(self.buffer_size, self.seq_len, self.patch_size)

        # --------------------------------------------------------------------------
        # Diffusion Decoder
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))
        decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=decoder_embed_dim, cross_attention_dim=encoder_embed_dim,
                                num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                                proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.diffdecoder = DiffDecoder(
            target_channels=self.vae_embed_dim,
            z_channels=encoder_embed_dim,
            width=decoder_embed_dim,
            diff_seqlen=self.seq_len * (self.img_size // 256) ** 2,
            num_sampling_steps=num_sampling_steps,
            blocks=decoder_blocks,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        # --------------------------------------------------------------------------
        self.initialize_weights()
        self.diffdecoder.net.initialize_weights()

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x, hw=16, patch_size=None):
        bsz, c, h, w = x.shape
        p = patch_size if patch_size else self.patch_size
        h_ = w_ = hw

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x, hw=16, patch_size=None):
        bsz = x.shape[0]
        p = patch_size if patch_size else self.patch_size
        c = self.vae_embed_dim
        h_ = w_ = hw

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def forward_encoder(self, x, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype), x], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x[:, self.buffer_size: self.buffer_size + 64, :] = self.mask_token

        # encoder position embedding
        if self.training:
            x = x.contiguous() + self.encoder_pos_embed_learned
        else:
            x = x.contiguous() + self.encoder_pos_embed_learned[:, :x.size(1), :]
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, self.attention_mask)
        else:
            for block in self.encoder_blocks:
                x = block(x, mask=self.attention_mask if self.training else self.attention_mask[:, :, :x.size(1), :x.size(1)])
        x = self.encoder_norm(x)

        x = x[:, self.buffer_size:]
        # x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1, 1)
        z = z.repeat(self.diffusion_batch_mul, 1, 1)
        loss = self.diffdecoder.forward(z=z, target=target)
        return loss

    def forward(self, img0, img1, img2, img3, labels):
        # class embed
        class_embedding = self.class_emb(labels)

        # patchify
        gt_latents0 = self.patchify(img0, hw=16, patch_size=1).clone().detach()  # [bsz, 256, 16]
        gt_latents1 = self.patchify(img1, hw=16, patch_size=1).clone().detach()  # [bsz, 256, 16]
        gt_latents2 = self.patchify(img2, hw=16, patch_size=1).clone().detach()  # [bsz, 256, 16]
        gt_latents3 = self.patchify(img3, hw=16, patch_size=1).clone().detach()  # [bsz, 256, 16]
        x0 = self.patchify(img0, hw=8, patch_size=2).clone()  # [bsz, 64, 16*4]
        x1 = self.patchify(img1, hw=8, patch_size=2).clone()  # [bsz, 64, 16*4]
        x2 = self.patchify(img2, hw=8, patch_size=2).clone()  # [bsz, 64, 16*4]

        x = torch.zeros((x0.size(0), self.seq_len, self.token_embed_dim), device=x0.device, dtype=x0.dtype)
        x[:, 64: 128, :] = x0
        x[:, 128: 192, :] = x1
        x[:, 192: 256, :] = x2
        # transformer encoder
        z = self.forward_encoder(x, class_embedding)
        z = z + self.diffusion_pos_embed_learned[:, :z.size(1), :]
        # diffusion deocder
        z0, z1, z2, z3 = z[:, : 64], z[:, : 128], z[:, : 192], z[:, : 256]
        loss0 = self.forward_loss(z=z0, target=gt_latents0)
        loss1 = self.forward_loss(z=z1, target=gt_latents1)
        loss2 = self.forward_loss(z=z2, target=gt_latents2)
        loss3 = self.forward_loss(z=z3, target=gt_latents3)
        loss = (loss0 + loss1 + loss2 + loss3) / 4
        return loss

    def samples(self, labels, num_sampling_steps=100, cfg=[1.0, 1.0, 1.0, 1.0], sampler='maruyama', scale_0=1.0, scale_1=1.0):
        bsz = labels.size(0)
        if type(cfg) == float:
            cfg = [cfg] * 4
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        class_embedding = self.class_emb(labels)
        noise = None
        for i in range(4):
            if i > 0:
                tokens[:, 64 * i: 64 * (i + 1), :] = self.patchify(tmp_tokens, hw=8, patch_size=2).clone()
            cur_tokens = tokens.clone()
            cur_class_embedding = class_embedding
            if cfg[i] != 1.0:
                cur_tokens = torch.cat([cur_tokens, cur_tokens], dim=0)
                cur_class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
            z = self.forward_encoder(cur_tokens[:, :64 * (i + 1), :], cur_class_embedding)
            z = z + self.diffusion_pos_embed_learned[:, :z.size(1), :]
            if i == 0 and noise is None:
                noise = randn_tensor((bsz, 256, self.vae_embed_dim), generator=None, device=z.device, dtype=z.dtype)

            if sampler == 'maruyama':
                sampled_token_latent = euler_maruyama_sampler(model=self.diffdecoder.net, latents=noise,
                                                              c=z[:, : 64 * (i + 1)],
                                                              num_steps=num_sampling_steps, cfg_scale=cfg[i],
                                                              scale_0=scale_0, scale_1=scale_1)
            else:
                # sampled_token_latent = self.diffdecoder.sample(z[:, : 64 * (i + 1)], noise=noise,
                #                                             num_sampling_steps=num_sampling_steps, cfg=cfg[i])
                raise Exception("sampler must be maruyama.")
            tmp_tokens = self.unpatchify(sampled_token_latent, hw=16, patch_size=1).to(torch.float32)

        res_tokens = tmp_tokens
        return res_tokens


def transdiff_base(**kwargs):
    model = Transdiff(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def transdiff_large(**kwargs):
    model = Transdiff(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def transdiff_huge(**kwargs):
    model = Transdiff(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

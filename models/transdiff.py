from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.diffdecoder import DiffDecoder
from models.crossattention import CrossAttentionBlock, AttentionBlock

from diffusers.utils.torch_utils import randn_tensor
from models.samplers import euler_maruyama_sampler


class Transdiff(nn.Module):
    def __init__(self, img_size=256, patch_size=1,
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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            AttentionBlock(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth * 2)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Diffusion Decoder
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))
        decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=decoder_embed_dim, cross_attention_dim=encoder_embed_dim,
                                num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                                proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.diffdecoder = DiffDecoder(
            target_channels=self.token_embed_dim,
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

    def forward_encoder(self, x, class_embedding, use_diversity_loss=False):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype), x],
                      dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x[:, self.buffer_size:] = self.mask_token

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        x = x[:, self.buffer_size:]
        if use_diversity_loss:
            loss_diversity = torch.matmul(x / x.norm(dim=2, keepdim=True),
                                          (x / x.norm(dim=2, keepdim=True)).permute(0, 2, 1)).mean()
            x = x + self.diffusion_pos_embed_learned
            return x, loss_diversity
        else:
            x = x + self.diffusion_pos_embed_learned
            return x, 0.0

    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1, 1)
        z = z.repeat(self.diffusion_batch_mul, 1, 1)
        loss = self.diffdecoder.forward(z=z, target=target)
        return loss

    def forward(self, imgs, labels):
        # class embed
        class_embedding = self.class_emb(labels)

        # patchify
        x = self.patchify(imgs, hw=(16 if self.img_size==256 else 32))
        gt_latents = x.clone().detach()

        x = torch.zeros((x.size(0), self.seq_len, self.token_embed_dim), device=x.device, dtype=x.dtype)
        # transformer encoder
        z, loss_diversity = self.forward_encoder(x, class_embedding)

        # diffusion deocder
        loss = self.forward_loss(z=z, target=gt_latents) + loss_diversity * 0.001
        return loss

    def samples(self, labels, num_sampling_steps=100, cfg=1.0, sampler='maruyama', scale_0=1.0, scale_1=1.0):
        bsz = labels.size(0)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        class_embedding = self.class_emb(labels)

        if not cfg == 1.0:
            tokens = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)

        z, loss_diversity = self.forward_encoder(tokens, class_embedding)

        noise = randn_tensor((bsz, self.seq_len * (self.img_size // 256) ** 2, self.token_embed_dim), generator=None,
                             device=z.device, dtype=z.dtype)
        if sampler == 'maruyama':
            sampled_token_latent = euler_maruyama_sampler(model=self.diffdecoder.net, latents=noise, c=z,
                                                          num_steps=num_sampling_steps, cfg_scale=cfg,
                                                          scale_0=scale_0, scale_1=scale_1)
        else:
            # sampled_token_latent = self.diffdecoder.sample(z, num_sampling_steps=num_sampling_steps, noise=noise, cfg=cfg)
            raise Exception("sampler must be maruyama.")
        tokens = self.unpatchify(sampled_token_latent, hw=16 * (self.img_size // 256))
        return tokens


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

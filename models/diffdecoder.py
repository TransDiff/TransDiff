import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from scipy.stats import norm
import numpy as np


class DiffDecoder(nn.Module):
    def __init__(self, target_channels, z_channels, diff_seqlen, width, num_sampling_steps, blocks,
                 grad_checkpointing=False):
        super(DiffDecoder, self).__init__()
        self.in_channels = target_channels
        self.net = DiffusionDeocder(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            diff_seqlen=diff_seqlen,
            blocks=blocks,
            grad_checkpointing=grad_checkpointing,
        )
        # self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('scheduler/')
        self.num_sampling_steps = int(num_sampling_steps)

    def sample_logit_normal(self, mu=0, sigma=1, size=1):
        # Generate samples from the normal distribution
        # & Transform samples to be in the range (0, 1) using the logistic function
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        samples = 1 / (1 + np.exp(-samples))
        samples = torch.tensor(samples, dtype=torch.float32)
        return samples

    def forward(self, target, z):
        noises = torch.concat([torch.randn_like(target[:1, ...]) for _ in range(target.size(0))], dim=0)
        timestep = (self.sample_logit_normal(size=target.shape[0]) * 1000).to(target.device, torch.long)
        t = timestep.view(-1, 1, 1)
        model_input = (1 - t / 1000) * target + (t / 1000) * noises
        model_target = noises - target
        model_output = self.net(model_input, z, timestep)
        loss = ((model_output - model_target) ** 2).mean()
        return loss.mean()

    # def sample(self, z, noise, num_sampling_steps=None, cfg=1.0):
    #     self.scheduler.set_timesteps(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps,
    #                                  device='cuda')
    #     timesteps = self.scheduler.timesteps
    #     latents = noise
    #     for i, t in enumerate(timesteps):
    #         latent_model_input = torch.cat([latents] * 2) if cfg > 1.0 else latents
    #         timestep = t.expand(latent_model_input.shape[0])
    #         noise_pred = self.net(x=latent_model_input, c=z, t=timestep)
    #         if cfg > 1.0:
    #             noise_pred, noise_uncond = noise_pred.chunk(2)
    #             noise_pred = noise_uncond + cfg * (noise_pred - noise_uncond)
    #         latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    #     return latents


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(hidden_size, hidden_size * 6)

    def forward(self, timestep):
        timestep = self.timesteps_proj(timestep)
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb).to(torch.bfloat16))
        return temb, timestep_proj


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, model_channels) / model_channels ** 0.5)

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table + t.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        # b, s, h = x.shape
        # x = x.reshape(b, s, 2, h//2).reshape(b, s*2, h//2)
        return x


class DiffusionDeocder(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            z_channels,
            blocks,
            diff_seqlen,
            grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = blocks
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, diff_seqlen, model_channels))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c, t):
        x = self.input_proj(x)
        x = x + self.diffusion_pos_embed_learned
        temb, timestep_proj = self.time_embed(t)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        c = self.cond_embed(c)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, c, timestep_proj)
        else:
            for block in self.res_blocks:
                x = block(x, c, timestep_proj)

        return self.final_layer(x, temb)

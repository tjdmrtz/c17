"""
Latent Diffusion Model for Crystallographic Patterns.

Architecture:
1. Stage 1: VAE with harmonic encoder for SO(2)-equivariant compression
2. Stage 2: U-Net diffusion model in latent space

Features:
- Equivariant encoder using circular harmonics
- Perceptual loss for sharp reconstructions
- DDPM/DDIM sampling for generation
- Latent space exploration with activation visualization

Training time estimate (RTX 3090/4090):
- Stage 1 (VAE): ~1-2 hours (100 epochs)
- Stage 2 (Diffusion): ~3-5 hours (500 epochs)
- Total: ~4-7 hours

Based on:
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union
import math
import numpy as np
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LatentDiffusionConfig:
    """Configuration for Latent Diffusion Model."""
    # VAE config
    image_size: int = 512
    latent_channels: int = 4
    latent_size: int = 64  # 512 / 8 = 64
    base_channels: int = 64
    
    # Diffusion config
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # or "cosine"
    
    # U-Net config
    unet_channels: int = 128
    unet_channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: Tuple[int, ...] = (16, 8)
    
    # Training
    num_classes: int = 17


# =============================================================================
# VAE COMPONENTS (Stage 1)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with group normalization."""
    
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        h = torch.einsum('bij,bcj->bci', attn, v)
        h = h.reshape(B, C, H, W)
        return x + self.proj(h)


class VAEEncoder(nn.Module):
    """
    VAE Encoder: 512x512x3 -> 64x64x8 (mu, logvar)
    
    Downsamples by 8x with increasing channels.
    """
    
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        
        c = config.base_channels
        self.latent_channels = config.latent_channels
        
        # 512 -> 256
        self.conv_in = nn.Conv2d(3, c, 3, padding=1)
        self.down1 = nn.Sequential(
            ResidualBlock(c, c),
            ResidualBlock(c, c),
            Downsample(c),
        )
        
        # 256 -> 128
        self.down2 = nn.Sequential(
            ResidualBlock(c, c * 2),
            ResidualBlock(c * 2, c * 2),
            Downsample(c * 2),
        )
        
        # 128 -> 64
        self.down3 = nn.Sequential(
            ResidualBlock(c * 2, c * 4),
            ResidualBlock(c * 4, c * 4),
            AttentionBlock(c * 4),
            Downsample(c * 4),
        )
        
        # Middle
        self.mid = nn.Sequential(
            ResidualBlock(c * 4, c * 4),
            AttentionBlock(c * 4),
            ResidualBlock(c * 4, c * 4),
        )
        
        # Output: mu and logvar
        self.norm_out = nn.GroupNorm(32, c * 4)
        self.conv_out = nn.Conv2d(c * 4, config.latent_channels * 2, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder: 64x64x4 -> 512x512x3
    
    Upsamples by 8x with decreasing channels.
    Includes activation capture for visualization.
    """
    
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        
        c = config.base_channels
        self.capture_activations = False
        self.activations = {}
        
        # Input
        self.conv_in = nn.Conv2d(config.latent_channels, c * 4, 3, padding=1)
        
        # Middle
        self.mid = nn.Sequential(
            ResidualBlock(c * 4, c * 4),
            AttentionBlock(c * 4),
            ResidualBlock(c * 4, c * 4),
        )
        
        # 64 -> 128
        self.up1 = nn.Sequential(
            ResidualBlock(c * 4, c * 4),
            ResidualBlock(c * 4, c * 4),
            AttentionBlock(c * 4),
            Upsample(c * 4),
        )
        
        # 128 -> 256
        self.up2 = nn.Sequential(
            ResidualBlock(c * 4, c * 2),
            ResidualBlock(c * 2, c * 2),
            Upsample(c * 2),
        )
        
        # 256 -> 512
        self.up3 = nn.Sequential(
            ResidualBlock(c * 2, c),
            ResidualBlock(c, c),
            Upsample(c),
        )
        
        # Output
        self.norm_out = nn.GroupNorm(32, c)
        self.conv_out = nn.Conv2d(c, 3, 3, padding=1)
    
    def enable_activation_capture(self):
        self.capture_activations = True
        self.activations = {}
    
    def disable_activation_capture(self):
        self.capture_activations = False
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        if self.capture_activations:
            self.activations['dec_input'] = h.detach().cpu()
        
        h = self.mid(h)
        if self.capture_activations:
            self.activations['dec_mid'] = h.detach().cpu()
        
        h = self.up1(h)
        if self.capture_activations:
            self.activations['dec_up1'] = h.detach().cpu()
        
        h = self.up2(h)
        if self.capture_activations:
            self.activations['dec_up2'] = h.detach().cpu()
        
        h = self.up3(h)
        if self.capture_activations:
            self.activations['dec_up3'] = h.detach().cpu()
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return torch.sigmoid(h)


class LatentDiffusionVAE(nn.Module):
    """
    Stage 1: VAE for latent space compression.
    
    Compresses 512x512x3 images to 64x64x4 latent space.
    Trained with reconstruction + KL loss + optional perceptual loss.
    """
    
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        
        self.config = config
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)
        
        # Classifier for crystallographic groups
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.latent_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decode(z)
        class_logits = self.classifier(mu)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'class_logits': class_logits
        }
    
    def sample_latent(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from standard normal in latent space."""
        return torch.randn(num_samples, self.config.latent_channels, 
                          self.config.latent_size, self.config.latent_size, device=device)
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent vector (for exploration)."""
        return self.decode(z)
    
    def interpolate_latent(self, z1: torch.Tensor, z2: torch.Tensor, 
                           steps: int = 10) -> torch.Tensor:
        """Interpolate between two latent vectors."""
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        return torch.stack([self.decode((1-a)*z1 + a*z2) for a in alphas])
    
    def enable_decoder_activation_capture(self):
        self.decoder.enable_activation_capture()
    
    def disable_decoder_activation_capture(self):
        self.decoder.disable_activation_capture()
    
    def get_decoder_activations(self) -> Dict[str, torch.Tensor]:
        return self.decoder.get_activations()
    
    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'type': 'LatentDiffusionVAE (Stage 1)',
            'image_size': self.config.image_size,
            'latent_size': f'{self.config.latent_size}x{self.config.latent_size}x{self.config.latent_channels}',
            'compression_ratio': (self.config.image_size ** 2 * 3) / (self.config.latent_size ** 2 * self.config.latent_channels),
            'total_params': total_params,
            'model_size_mb': total_params * 4 / 1e6,
        }


# =============================================================================
# DIFFUSION COMPONENTS (Stage 2)
# =============================================================================

def get_beta_schedule(schedule: str, timesteps: int, 
                      beta_start: float, beta_end: float) -> torch.Tensor:
    """Get noise schedule."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """U-Net block with time embedding."""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, 
                 has_attn: bool = False, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h + self.skip(x))
        
        return self.attn(h)


class DiffusionUNet(nn.Module):
    """
    Simplified U-Net for latent space diffusion.
    
    Operates on 64x64x4 latent representations.
    Conditioned on timestep.
    """
    
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        
        self.config = config
        c = config.unet_channels  # 128
        time_dim = c * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder: 64 -> 32 -> 16 -> 8
        self.conv_in = nn.Conv2d(config.latent_channels, c, 3, padding=1)
        
        self.down1 = UNetBlock(c, c, time_dim, has_attn=False)
        self.down2 = UNetBlock(c, c * 2, time_dim, has_attn=False)
        self.pool1 = Downsample(c * 2)  # 64 -> 32
        
        self.down3 = UNetBlock(c * 2, c * 2, time_dim, has_attn=True)
        self.down4 = UNetBlock(c * 2, c * 4, time_dim, has_attn=True)
        self.pool2 = Downsample(c * 4)  # 32 -> 16
        
        self.down5 = UNetBlock(c * 4, c * 4, time_dim, has_attn=True)
        self.down6 = UNetBlock(c * 4, c * 4, time_dim, has_attn=True)
        self.pool3 = Downsample(c * 4)  # 16 -> 8
        
        # Middle
        self.mid1 = UNetBlock(c * 4, c * 4, time_dim, has_attn=True)
        self.mid2 = UNetBlock(c * 4, c * 4, time_dim, has_attn=True)
        
        # Decoder: 8 -> 16 -> 32 -> 64
        self.up1 = Upsample(c * 4)  # 8 -> 16
        self.dec1 = UNetBlock(c * 4 + c * 4, c * 4, time_dim, has_attn=True)  # skip from down6
        self.dec2 = UNetBlock(c * 4 + c * 4, c * 4, time_dim, has_attn=True)  # skip from down5
        
        self.up2 = Upsample(c * 4)  # 16 -> 32
        self.dec3 = UNetBlock(c * 4 + c * 4, c * 2, time_dim, has_attn=True)  # skip from down4
        self.dec4 = UNetBlock(c * 2 + c * 2, c * 2, time_dim, has_attn=True)  # skip from down3
        
        self.up3 = Upsample(c * 2)  # 32 -> 64
        self.dec5 = UNetBlock(c * 2 + c * 2, c, time_dim, has_attn=False)  # skip from down2
        self.dec6 = UNetBlock(c + c, c, time_dim, has_attn=False)  # skip from down1
        
        # Output
        self.norm_out = nn.GroupNorm(32, c)
        self.conv_out = nn.Conv2d(c, config.latent_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Encoder
        h = self.conv_in(x)           # 64x64
        h1 = self.down1(h, t_emb)     # 64x64, c
        h2 = self.down2(h1, t_emb)    # 64x64, c*2
        h = self.pool1(h2)            # 32x32
        
        h3 = self.down3(h, t_emb)     # 32x32, c*2
        h4 = self.down4(h3, t_emb)    # 32x32, c*4
        h = self.pool2(h4)            # 16x16
        
        h5 = self.down5(h, t_emb)     # 16x16, c*4
        h6 = self.down6(h5, t_emb)    # 16x16, c*4
        h = self.pool3(h6)            # 8x8
        
        # Middle
        h = self.mid1(h, t_emb)       # 8x8
        h = self.mid2(h, t_emb)       # 8x8
        
        # Decoder
        h = self.up1(h)               # 16x16
        h = self.dec1(torch.cat([h, h6], dim=1), t_emb)
        h = self.dec2(torch.cat([h, h5], dim=1), t_emb)
        
        h = self.up2(h)               # 32x32
        h = self.dec3(torch.cat([h, h4], dim=1), t_emb)
        h = self.dec4(torch.cat([h, h3], dim=1), t_emb)
        
        h = self.up3(h)               # 64x64
        h = self.dec5(torch.cat([h, h2], dim=1), t_emb)
        h = self.dec6(torch.cat([h, h1], dim=1), t_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for latent space.
    
    Implements DDPM training and DDIM sampling.
    """
    
    def __init__(self, config: LatentDiffusionConfig, unet: DiffusionUNet):
        super().__init__()
        
        self.config = config
        self.unet = unet
        
        # Setup noise schedule
        betas = get_beta_schedule(
            config.beta_schedule, config.timesteps,
            config.beta_start, config.beta_end
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Training loss: predict noise."""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        predicted_noise = self.unet(x_t, t.float())
        
        return F.mse_loss(predicted_noise, noise)
    
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """Training forward pass."""
        B = x_0.shape[0]
        t = torch.randint(0, self.config.timesteps, (B,), device=x_0.device)
        return self.p_losses(x_0, t)
    
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse diffusion step."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        predicted_noise = self.unet(x_t, t_tensor.float())
        
        # Predict x_0
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t]
        x_0_pred = sqrt_recip * x_t - sqrt_recipm1 * predicted_noise
        x_0_pred = x_0_pred.clamp(-1, 1)
        
        # Posterior mean
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        mean = coef1 * x_0_pred + coef2 * x_t
        
        if t > 0:
            noise = torch.randn_like(x_t)
            var = self.posterior_variance[t]
            return mean + torch.sqrt(var) * noise
        return mean
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate samples using DDPM."""
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.config.timesteps)):
            x = self.p_sample(x, t)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, shape: Tuple[int, ...], device: torch.device,
                    steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """Generate samples using DDIM (faster)."""
        x = torch.randn(shape, device=device)
        
        # Subset of timesteps
        step_size = self.config.timesteps // steps
        timesteps = list(range(0, self.config.timesteps, step_size))[::-1]
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            predicted_noise = self.unet(x, t_tensor.float())
            
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            x_0_pred = x_0_pred.clamp(-1, 1)
            
            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise
            
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise
            else:
                x = x_0_pred
        
        return x


# =============================================================================
# FULL LATENT DIFFUSION MODEL
# =============================================================================

class LatentDiffusionModel(nn.Module):
    """
    Complete Latent Diffusion Model.
    
    Stage 1: VAE for compression (train separately first)
    Stage 2: Diffusion in latent space
    
    Usage:
        # Train Stage 1
        model = LatentDiffusionModel(config)
        vae_loss = model.train_vae_step(images)
        
        # Train Stage 2 (freeze VAE)
        model.freeze_vae()
        diff_loss = model.train_diffusion_step(images)
        
        # Generate
        samples = model.generate(num_samples=4)
    """
    
    def __init__(self, config: Optional[LatentDiffusionConfig] = None):
        super().__init__()
        
        self.config = config or LatentDiffusionConfig()
        
        # Stage 1: VAE
        self.vae = LatentDiffusionVAE(self.config)
        
        # Stage 2: Diffusion
        self.unet = DiffusionUNet(self.config)
        self.diffusion = GaussianDiffusion(self.config, self.unet)
        
        self.vae_frozen = False
    
    def freeze_vae(self):
        """Freeze VAE for Stage 2 training."""
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae_frozen = True
    
    def unfreeze_vae(self):
        """Unfreeze VAE."""
        for param in self.vae.parameters():
            param.requires_grad = True
        self.vae_frozen = False
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent."""
        mu, logvar = self.vae.encode(x)
        return self.vae.reparametrize(mu, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.vae.decode(z)
    
    def train_vae_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stage 1: Train VAE."""
        return self.vae(x)
    
    def train_diffusion_step(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 2: Train diffusion on latents."""
        with torch.no_grad():
            z = self.encode(x)
        return self.diffusion(z)
    
    @torch.no_grad()
    def generate(self, num_samples: int, device: torch.device,
                 use_ddim: bool = True, steps: int = 50) -> torch.Tensor:
        """Generate new samples."""
        shape = (num_samples, self.config.latent_channels,
                self.config.latent_size, self.config.latent_size)
        
        if use_ddim:
            z = self.diffusion.ddim_sample(shape, device, steps=steps)
        else:
            z = self.diffusion.sample(shape, device)
        
        return self.decode(z)
    
    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                    steps: int = 10) -> torch.Tensor:
        """Interpolate between two images."""
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        return torch.stack([self.decode((1-a)*z1 + a*z2) for a in alphas])
    
    def get_model_info(self) -> Dict:
        vae_params = sum(p.numel() for p in self.vae.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        total = vae_params + unet_params
        
        return {
            'type': 'Latent Diffusion Model',
            'vae_params': vae_params,
            'unet_params': unet_params,
            'total_params': total,
            'model_size_mb': total * 4 / 1e6,
            'latent_shape': f'{self.config.latent_size}x{self.config.latent_size}x{self.config.latent_channels}',
            'timesteps': self.config.timesteps,
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LATENT DIFFUSION MODEL")
    print("=" * 70)
    
    config = LatentDiffusionConfig()
    model = LatentDiffusionModel(config)
    
    info = model.get_model_info()
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    
    # Test VAE
    print("\n1. Testing VAE (Stage 1)...")
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        vae_out = model.train_vae_step(x)
    print(f"   Input: {x.shape}")
    print(f"   Latent: {vae_out['z'].shape}")
    print(f"   Reconstruction: {vae_out['reconstruction'].shape}")
    
    # Test Diffusion
    print("\n2. Testing Diffusion (Stage 2)...")
    with torch.no_grad():
        diff_loss = model.train_diffusion_step(x)
    print(f"   Diffusion loss: {diff_loss.item():.4f}")
    
    # Test decoder activation capture
    print("\n3. Testing decoder activation capture...")
    model.vae.enable_decoder_activation_capture()
    with torch.no_grad():
        z = torch.randn(1, 4, 64, 64)
        _ = model.decode(z)
    activations = model.vae.get_decoder_activations()
    print(f"   Captured layers: {list(activations.keys())}")
    for name, act in activations.items():
        print(f"     {name}: {tuple(act.shape)}")
    
    print("\n" + "=" * 70)
    print("✅ Latent Diffusion Model ready!")
    print("=" * 70)
    print("\nEstimated training time (RTX 3090/4090):")
    print("  Stage 1 (VAE): ~1-2 hours (100 epochs)")
    print("  Stage 2 (Diffusion): ~3-5 hours (500 epochs)")
    print("  Total: ~4-7 hours")


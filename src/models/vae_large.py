"""
Large Variational Autoencoder for Crystallographic Patterns.

This is a more powerful VAE architecture with:
- Deeper encoder/decoder with more residual blocks
- Attention mechanisms for better pattern capture
- Skip connections for improved gradient flow
- Larger channel dimensions

Designed for larger datasets (10k+ samples).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class SelfAttention2d(nn.Module):
    """Self-attention layer for 2D feature maps."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.size()
        
        # Queries, keys, values
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)
        
        # Attention
        attn = F.softmax(torch.bmm(q, k) / math.sqrt(c // 8), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        
        return self.gamma * out + x


class ResidualBlockLarge(nn.Module):
    """Enhanced residual block with optional attention."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 downsample: bool = False, upsample: bool = False,
                 use_attention: bool = False):
        super().__init__()
        
        self.downsample = downsample
        self.upsample = upsample
        
        stride = 2 if downsample else 1
        
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels or downsample or upsample:
            if upsample:
                self.shortcut = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
        
        # Optional attention
        self.attention = SelfAttention2d(out_channels) if use_attention else None
        
        # Squeeze-and-excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x_up = self.up(x)
            out = F.relu(self.bn1(self.conv1(x_up)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        # Self-attention
        if self.attention is not None:
            out = self.attention(out)
        
        # Residual
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out


class EncoderLarge(nn.Module):
    """Large encoder with attention and deep residual blocks."""
    
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 256,
                 base_channels: int = 64,
                 input_size: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        c = base_channels
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # Residual stages with progressive downsampling
        # 64 -> 32 -> 16 -> 8 -> 4
        self.stage1 = nn.Sequential(
            ResidualBlockLarge(c, c * 2, downsample=True),
            ResidualBlockLarge(c * 2, c * 2),
        )
        
        self.stage2 = nn.Sequential(
            ResidualBlockLarge(c * 2, c * 4, downsample=True),
            ResidualBlockLarge(c * 4, c * 4),
        )
        
        self.stage3 = nn.Sequential(
            ResidualBlockLarge(c * 4, c * 8, downsample=True, use_attention=True),
            ResidualBlockLarge(c * 8, c * 8),
        )
        
        self.stage4 = nn.Sequential(
            ResidualBlockLarge(c * 8, c * 8, downsample=True, use_attention=True),
            ResidualBlockLarge(c * 8, c * 8),
        )
        
        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(c * 8, latent_dim)
        self.fc_logvar = nn.Linear(c * 8, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class DecoderLarge(nn.Module):
    """Large decoder with attention and skip-connection ready."""
    
    def __init__(self,
                 out_channels: int = 1,
                 latent_dim: int = 256,
                 base_channels: int = 64,
                 output_size: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        c = base_channels
        
        # Project latent to spatial
        self.fc = nn.Linear(latent_dim, c * 8 * 4 * 4)
        self.initial_size = 4
        self.initial_channels = c * 8
        
        # Upsampling stages
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.stage1 = nn.Sequential(
            ResidualBlockLarge(c * 8, c * 8, upsample=True, use_attention=True),
            ResidualBlockLarge(c * 8, c * 8),
        )
        
        self.stage2 = nn.Sequential(
            ResidualBlockLarge(c * 8, c * 4, upsample=True, use_attention=True),
            ResidualBlockLarge(c * 4, c * 4),
        )
        
        self.stage3 = nn.Sequential(
            ResidualBlockLarge(c * 4, c * 2, upsample=True),
            ResidualBlockLarge(c * 2, c * 2),
        )
        
        self.stage4 = nn.Sequential(
            ResidualBlockLarge(c * 2, c, upsample=True),
            ResidualBlockLarge(c, c),
        )
        
        self.stage5 = nn.Sequential(
            ResidualBlockLarge(c, c // 2, upsample=True),
            ResidualBlockLarge(c // 2, c // 2),
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(c // 2, c // 2, 3, padding=1),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        
        x = self.stage1(x)   # 4 -> 8
        x = self.stage2(x)   # 8 -> 16
        x = self.stage3(x)   # 16 -> 32
        x = self.stage4(x)   # 32 -> 64
        x = self.stage5(x)   # 64 -> 128
        
        x = self.output(x)
        
        return x


class CrystallographicVAELarge(nn.Module):
    """
    Large VAE for crystallographic patterns.
    
    Features:
    - Deep residual encoder/decoder
    - Self-attention at bottleneck
    - Squeeze-and-excitation blocks
    - ~15M parameters (vs ~3.5M for base model)
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 256,
                 base_channels: int = 64,
                 input_size: int = 128,
                 num_classes: int = 17):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.encoder = EncoderLarge(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            input_size=input_size
        )
        
        self.decoder = DecoderLarge(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_size=input_size
        )
        
        # Classifier from latent
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, num_classes)
        )
        
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
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
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        interpolations = []
        
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            interpolations.append(self.decode(z))
        
        return torch.cat(interpolations, dim=0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = CrystallographicVAELarge(latent_dim=256, base_channels=64)
    print(f"Large VAE parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 1, 128, 128)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"Latent shape: {outputs['mu'].shape}")


"""
Medium-sized VAE for Crystallographic Patterns.

Balanced architecture:
- ~6M parameters (between base 3.5M and large 15M)
- Some attention mechanisms
- Good for datasets of 20k-50k samples

Model sizing guidelines:
- 5k-15k samples → VAE Base (3.5M params)
- 15k-50k samples → VAE Medium (6M params) 
- 50k+ samples → VAE Large (15M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b, c)
        se = self.excitation(se).view(b, c, 1, 1)
        return x * se


class ResBlockMedium(nn.Module):
    """Residual block with SE attention."""
    
    def __init__(self, in_ch: int, out_ch: int, 
                 downsample: bool = False, upsample: bool = False):
        super().__init__()
        
        self.upsample = upsample
        stride = 2 if downsample else 1
        
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.se = SEBlock(out_ch)
        
        # Shortcut
        if in_ch != out_ch or downsample or upsample:
            if upsample:
                self.shortcut = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.BatchNorm2d(out_ch)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                    nn.BatchNorm2d(out_ch)
                )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x_up = self.up(x)
            out = F.relu(self.bn1(self.conv1(x_up)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + self.shortcut(x))
        
        return out


class EncoderMedium(nn.Module):
    """Medium encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4"""
    
    def __init__(self, in_channels: int = 1, latent_dim: int = 192, 
                 base_channels: int = 48):
        super().__init__()
        
        c = base_channels  # 48
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # 64 -> 32
        self.stage1 = ResBlockMedium(c, c * 2, downsample=True)
        # 32 -> 16
        self.stage2 = ResBlockMedium(c * 2, c * 4, downsample=True)
        # 16 -> 8
        self.stage3 = ResBlockMedium(c * 4, c * 6, downsample=True)
        # 8 -> 4
        self.stage4 = ResBlockMedium(c * 6, c * 8, downsample=True)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(c * 8, latent_dim)
        self.fc_logvar = nn.Linear(c * 8, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)


class DecoderMedium(nn.Module):
    """Medium decoder: 4 -> 8 -> 16 -> 32 -> 64 -> 128"""
    
    def __init__(self, out_channels: int = 1, latent_dim: int = 192,
                 base_channels: int = 48):
        super().__init__()
        
        c = base_channels
        self.c = c
        
        self.fc = nn.Linear(latent_dim, c * 8 * 4 * 4)
        
        # 4 -> 8
        self.stage1 = ResBlockMedium(c * 8, c * 6, upsample=True)
        # 8 -> 16
        self.stage2 = ResBlockMedium(c * 6, c * 4, upsample=True)
        # 16 -> 32
        self.stage3 = ResBlockMedium(c * 4, c * 2, upsample=True)
        # 32 -> 64
        self.stage4 = ResBlockMedium(c * 2, c, upsample=True)
        # 64 -> 128
        self.stage5 = ResBlockMedium(c, c // 2, upsample=True)
        
        self.output = nn.Sequential(
            nn.Conv2d(c // 2, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.c * 8, 4, 4)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.output(x)


class CrystallographicVAEMedium(nn.Module):
    """
    Medium VAE: ~6M parameters.
    
    Recommended for datasets of 15k-50k samples.
    Good balance between capacity and regularization.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 192,
                 base_channels: int = 48,
                 input_size: int = 128,
                 num_classes: int = 17):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.encoder = EncoderMedium(in_channels, latent_dim, base_channels)
        self.decoder = DecoderMedium(in_channels, latent_dim, base_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
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
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                    steps: int = 10) -> torch.Tensor:
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        return torch.cat([self.decode((1-a)*mu1 + a*mu2) for a in alphas], dim=0)


def get_model_for_dataset_size(num_samples: int, resolution: int = 128):
    """
    Get appropriate model based on dataset size.
    
    Returns model class and recommended hyperparameters.
    """
    from .vae import CrystallographicVAE
    from .vae_large import CrystallographicVAELarge
    
    if num_samples < 15000:
        return {
            'model_class': CrystallographicVAE,
            'latent_dim': 128,
            'base_channels': 32,
            'params': '~3.5M',
            'recommended_batch_size': 32,
            'recommended_lr': 1e-3,
        }
    elif num_samples < 50000:
        return {
            'model_class': CrystallographicVAEMedium,
            'latent_dim': 192,
            'base_channels': 48,
            'params': '~6M',
            'recommended_batch_size': 48,
            'recommended_lr': 5e-4,
        }
    else:
        return {
            'model_class': CrystallographicVAELarge,
            'latent_dim': 256,
            'base_channels': 64,
            'params': '~15M',
            'recommended_batch_size': 64,
            'recommended_lr': 3e-4,
        }


if __name__ == "__main__":
    model = CrystallographicVAEMedium()
    params = sum(p.numel() for p in model.parameters())
    print(f"Medium VAE parameters: {params:,}")
    
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    print(f"Input: {x.shape} -> Recon: {out['reconstruction'].shape}, Latent: {out['mu'].shape}")








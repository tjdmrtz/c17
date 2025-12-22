#!/usr/bin/env python3
"""
Simple and Effective RGB VAE for Crystallographic Patterns.

No complex symmetry operations - just a clean, working VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class SimpleVAEConfig:
    """Configuration for Simple VAE."""
    resolution: int = 256
    in_channels: int = 3
    latent_dim: int = 64  # Larger latent space for better reconstruction
    hidden_dims: Tuple[int, ...] = (32, 64, 128, 256, 512)
    beta: float = 0.001  # Very low KL weight initially
    

class ResBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Encoder(nn.Module):
    """Simple convolutional encoder."""
    
    def __init__(self, config: SimpleVAEConfig):
        super().__init__()
        self.config = config
        
        # Build encoder layers
        layers = []
        in_ch = config.in_channels
        
        for h_dim in config.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
                ResBlock(h_dim),
            ))
            in_ch = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # After 5 downsamples: 256 -> 8
        self.flat_size = config.hidden_dims[-1] * 8 * 8
        
        # Latent projections
        self.fc_mu = nn.Linear(self.flat_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, config.latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Simple convolutional decoder."""
    
    def __init__(self, config: SimpleVAEConfig):
        super().__init__()
        self.config = config
        
        # Project from latent to spatial
        self.flat_size = config.hidden_dims[-1] * 8 * 8
        self.fc = nn.Linear(config.latent_dim, self.flat_size)
        
        # Build decoder layers (reverse of encoder)
        layers = []
        hidden_dims_rev = list(reversed(config.hidden_dims))
        
        for i in range(len(hidden_dims_rev) - 1):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_rev[i], hidden_dims_rev[i+1], 
                                   4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims_rev[i+1]),
                nn.LeakyReLU(0.2),
                ResBlock(hidden_dims_rev[i+1]),
            ))
        
        # Final layer to RGB
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims_rev[-1], hidden_dims_rev[-1], 
                               4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims_rev[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims_rev[-1], config.in_channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        ))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.config.hidden_dims[-1], 8, 8)
        x = self.decoder(x)
        return x


class SimpleVAE(nn.Module):
    """
    Simple VAE for RGB crystallographic patterns.
    
    No complex symmetry operations - focuses on good reconstruction.
    """
    
    def __init__(self, config: SimpleVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with clamping for stability."""
        logvar = torch.clamp(logvar, min=-20, max=2)  # Prevent extreme values
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, group_name: str = None) -> Dict[str, torch.Tensor]:
        """Forward pass (group_name ignored for compatibility)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }
    
    def encode(self, x: torch.Tensor, group_name: str = None) -> torch.Tensor:
        """Encode to latent space (returns mu)."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor, group_name: str = None) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from prior."""
        z = torch.randn(n_samples, self.config.latent_dim, device=device)
        return self.decoder(z)


class SimpleVAELoss(nn.Module):
    """Loss function for Simple VAE with beta scheduling."""
    
    def __init__(self, config: SimpleVAEConfig):
        super().__init__()
        self.config = config
        self.beta = config.beta
        
    def forward(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], 
                group_name: str = None) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with numerical stability."""
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-20, max=2)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence with stability
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_loss, min=0, max=100)  # Prevent negative or huge KL
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss
        
        # Check for NaN and replace with safe value
        if torch.isnan(loss):
            loss = recon_loss  # Fall back to just recon loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
    
    def set_beta(self, beta: float):
        """Update beta (for scheduling)."""
        self.beta = beta


def create_simple_vae(latent_dim: int = 64, resolution: int = 256) -> Tuple[SimpleVAE, SimpleVAEConfig]:
    """Create a simple VAE with good defaults."""
    config = SimpleVAEConfig(
        resolution=resolution,
        latent_dim=latent_dim,
        beta=0.0001,  # Start with very low KL weight
    )
    model = SimpleVAE(config)
    return model, config


if __name__ == "__main__":
    # Test
    config = SimpleVAEConfig(latent_dim=64)
    model = SimpleVAE(config)
    
    x = torch.randn(4, 3, 256, 256)
    out = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Recon: {out['recon'].shape}")
    print(f"Latent: {out['z'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


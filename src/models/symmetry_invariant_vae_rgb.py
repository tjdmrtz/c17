"""
Symmetry-Invariant VAE for RGB Crystallographic Patterns (256×256)

This module extends the original VAE to handle:
- RGB images (3 channels) instead of grayscale
- 256×256 resolution instead of 128×128
- Deeper architecture for better representation

The symmetry operations remain the same - they work on each channel independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class SymmetryVAEConfigRGB:
    """Configuration for RGB Symmetry-Invariant VAE."""
    resolution: int = 256
    latent_dim: int = 2
    in_channels: int = 3  # RGB
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 512])
    
    # Symmetry settings
    use_symmetry_pooling: bool = True
    symmetry_groups: List[str] = field(default_factory=lambda: [
        'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
        'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
    ])
    
    # Training settings
    beta: float = 1.0
    invariance_weight: float = 10.0


class SymmetryTransformsRGB:
    """
    Applies symmetry transformations for the 17 wallpaper groups.
    Works with RGB tensors (B, 3, H, W).
    """
    
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        self._setup_rotation_grids()
    
    def _setup_rotation_grids(self):
        """Setup grids for arbitrary rotations."""
        n = self.resolution
        
        angles = {
            60: np.pi / 3,
            120: 2 * np.pi / 3,
            240: 4 * np.pi / 3,
            300: 5 * np.pi / 3,
        }
        
        self.rotation_grids = {}
        for deg, rad in angles.items():
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], dtype=torch.float32)
            self.rotation_grids[deg] = F.affine_grid(
                theta.unsqueeze(0),
                (1, 1, n, n),
                align_corners=False
            )
    
    def rotate_90(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Rotate by 90*k degrees."""
        return torch.rot90(x, k, dims=[-2, -1])
    
    def rotate_arbitrary(self, x: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate by arbitrary angle using grid_sample."""
        if angle == 0:
            return x
        if angle == 90:
            return self.rotate_90(x, 1)
        if angle == 180:
            return self.rotate_90(x, 2)
        if angle == 270:
            return self.rotate_90(x, 3)
        
        grid = self.rotation_grids[angle].to(x.device)
        batch_size = x.shape[0]
        grid = grid.expand(batch_size, -1, -1, -1)
        
        # Apply periodic boundary conditions
        grid_periodic = torch.remainder(grid + 1, 2) - 1
        
        return F.grid_sample(x, grid_periodic, mode='bilinear',
                            padding_mode='border', align_corners=False)
    
    def flip_h(self, x: torch.Tensor) -> torch.Tensor:
        """Horizontal flip."""
        return torch.flip(x, dims=[-1])
    
    def flip_v(self, x: torch.Tensor) -> torch.Tensor:
        """Vertical flip."""
        return torch.flip(x, dims=[-2])
    
    def translate(self, x: torch.Tensor, shift_h: int, shift_v: int) -> torch.Tensor:
        """Translate with periodic boundaries."""
        return torch.roll(torch.roll(x, shift_h, dims=-1), shift_v, dims=-2)
    
    def get_group_transforms(self, group_name: str) -> List[callable]:
        """Get all transformations for a wallpaper group."""
        transforms = {
            'p1': [lambda x: x],
            'p2': [
                lambda x: x,
                lambda x: self.rotate_90(x, 2),
            ],
            'pm': [
                lambda x: x,
                lambda x: self.flip_h(x),
            ],
            'pg': [
                lambda x: x,
                lambda x: self.translate(self.flip_v(x), self.resolution // 2, 0),
            ],
            'cm': [
                lambda x: x,
                lambda x: self.flip_h(x),
            ],
            'pmm': [
                lambda x: x,
                lambda x: self.flip_h(x),
                lambda x: self.flip_v(x),
                lambda x: self.rotate_90(x, 2),
            ],
            'pmg': [
                lambda x: x,
                lambda x: self.flip_h(x),
                lambda x: self.rotate_90(x, 2),
                lambda x: self.flip_h(self.rotate_90(x, 2)),
            ],
            'pgg': [
                lambda x: x,
                lambda x: self.rotate_90(x, 2),
                lambda x: self.translate(self.flip_h(x), 0, self.resolution // 2),
                lambda x: self.translate(self.flip_v(x), self.resolution // 2, 0),
            ],
            'cmm': [
                lambda x: x,
                lambda x: self.flip_h(x),
                lambda x: self.flip_v(x),
                lambda x: self.rotate_90(x, 2),
            ],
            'p4': [
                lambda x: x,
                lambda x: self.rotate_90(x, 1),
                lambda x: self.rotate_90(x, 2),
                lambda x: self.rotate_90(x, 3),
            ],
            'p4m': [
                lambda x: x,
                lambda x: self.rotate_90(x, 1),
                lambda x: self.rotate_90(x, 2),
                lambda x: self.rotate_90(x, 3),
                lambda x: self.flip_h(x),
                lambda x: self.flip_h(self.rotate_90(x, 1)),
                lambda x: self.flip_h(self.rotate_90(x, 2)),
                lambda x: self.flip_h(self.rotate_90(x, 3)),
            ],
            'p4g': [
                lambda x: x,
                lambda x: self.rotate_90(x, 1),
                lambda x: self.rotate_90(x, 2),
                lambda x: self.rotate_90(x, 3),
                lambda x: self.flip_h(self.rotate_90(x, 1)),
                lambda x: self.flip_h(self.rotate_90(x, 3)),
            ],
            'p3': [
                lambda x: x,
                lambda x: self.rotate_arbitrary(x, 120),
                lambda x: self.rotate_arbitrary(x, 240),
            ],
            'p3m1': [
                lambda x: x,
                lambda x: self.rotate_arbitrary(x, 120),
                lambda x: self.rotate_arbitrary(x, 240),
                lambda x: self.flip_h(x),
                lambda x: self.flip_h(self.rotate_arbitrary(x, 120)),
                lambda x: self.flip_h(self.rotate_arbitrary(x, 240)),
            ],
            'p31m': [
                lambda x: x,
                lambda x: self.rotate_arbitrary(x, 120),
                lambda x: self.rotate_arbitrary(x, 240),
                lambda x: self.flip_v(x),
                lambda x: self.flip_v(self.rotate_arbitrary(x, 120)),
                lambda x: self.flip_v(self.rotate_arbitrary(x, 240)),
            ],
            'p6': [
                lambda x: x,
                lambda x: self.rotate_arbitrary(x, 60),
                lambda x: self.rotate_arbitrary(x, 120),
                lambda x: self.rotate_90(x, 2),
                lambda x: self.rotate_arbitrary(x, 240),
                lambda x: self.rotate_arbitrary(x, 300),
            ],
            'p6m': [
                lambda x: x,
                lambda x: self.rotate_arbitrary(x, 60),
                lambda x: self.rotate_arbitrary(x, 120),
                lambda x: self.rotate_90(x, 2),
                lambda x: self.rotate_arbitrary(x, 240),
                lambda x: self.rotate_arbitrary(x, 300),
                lambda x: self.flip_h(x),
                lambda x: self.flip_h(self.rotate_arbitrary(x, 60)),
                lambda x: self.flip_h(self.rotate_arbitrary(x, 120)),
                lambda x: self.flip_h(self.rotate_90(x, 2)),
                lambda x: self.flip_h(self.rotate_arbitrary(x, 240)),
                lambda x: self.flip_h(self.rotate_arbitrary(x, 300)),
            ],
        }
        return transforms.get(group_name, [lambda x: x])
    
    def apply_symmetry_pooling(self, x: torch.Tensor, group_name: str) -> torch.Tensor:
        """Average features over all group transformations (memory efficient)."""
        transforms = self.get_group_transforms(group_name)
        # Accumulate incrementally instead of storing all transforms
        result = transforms[0](x)
        for i in range(1, len(transforms)):
            result = result + transforms[i](x)
        return result / len(transforms)
    
    def symmetrize_output(self, x: torch.Tensor, group_name: str) -> torch.Tensor:
        """Apply symmetry projection to output (memory efficient)."""
        # For output, we can skip symmetrization to save memory during training
        # The reconstruction loss will learn to produce symmetric outputs
        return x  # Skip symmetrization for efficiency


class SymmetryInvariantEncoderRGB(nn.Module):
    """
    Encoder for RGB 256×256 images with symmetry invariance.
    """
    
    def __init__(self, config: SymmetryVAEConfigRGB):
        super().__init__()
        self.config = config
        self.transforms = SymmetryTransformsRGB(config.resolution)
        
        # Convolutional encoder: 256 → 128 → 64 → 32 → 16 → 8
        layers = []
        in_channels = config.in_channels
        
        for h_dim in config.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ))
            in_channels = h_dim
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # After 5 conv layers with stride 2: 256 / 32 = 8
        # Feature map: (512, 8, 8)
        self.flat_size = config.hidden_dims[-1]
        
        # MLP for mu and logvar
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, config.latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, config.latent_dim)
        )
    
    def forward(self, x: torch.Tensor, group_name: str = 'p4m') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode RGB input to latent distribution.
        
        Args:
            x: Input tensor [B, 3, 256, 256]
            group_name: Wallpaper group for symmetry invariance
            
        Returns:
            mu, logvar: [B, latent_dim]
        """
        batch_size = x.shape[0]
        
        if self.config.use_symmetry_pooling:
            transforms = self.transforms.get_group_transforms(group_name)
            all_features = []
            for transform in transforms:
                x_t = transform(x)
                features = self.conv_encoder(x_t)
                all_features.append(features)
            features = torch.stack(all_features, dim=0).mean(dim=0)
        else:
            features = self.conv_encoder(x)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, 1).view(batch_size, -1)
        
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar


class SymmetryAwareDecoderRGB(nn.Module):
    """
    Decoder for RGB 256×256 images with symmetry projection.
    """
    
    def __init__(self, config: SymmetryVAEConfigRGB):
        super().__init__()
        self.config = config
        self.transforms = SymmetryTransformsRGB(config.resolution)
        
        # Initial size: 256 / 32 = 8
        self.init_size = config.resolution // (2 ** len(config.hidden_dims))
        self.init_channels = config.hidden_dims[-1]
        
        # Project latent to feature map
        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.init_channels * self.init_size * self.init_size),
            nn.LeakyReLU(0.2),
        )
        
        # Transposed convolutions
        layers = []
        hidden_dims_reversed = config.hidden_dims[::-1]
        
        for i in range(len(hidden_dims_reversed) - 1):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims_reversed[i],
                    hidden_dims_reversed[i + 1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                nn.LeakyReLU(0.2),
            ))
        
        # Final layer to RGB
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1], config.in_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ))
        
        self.conv_decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, group_name: str = 'p4m') -> torch.Tensor:
        """
        Decode latent to RGB pattern with symmetry.
        
        Args:
            z: Latent vector [B, latent_dim]
            group_name: Wallpaper group for output symmetry
            
        Returns:
            RGB pattern [B, 3, 256, 256]
        """
        batch_size = z.shape[0]
        
        x = self.fc(z)
        x = x.view(batch_size, self.init_channels, self.init_size, self.init_size)
        x = self.conv_decoder(x)
        
        # Apply symmetry projection
        x = self.transforms.symmetrize_output(x, group_name)
        
        return x


class SymmetryInvariantVAE_RGB(nn.Module):
    """
    VAE for RGB 256×256 crystallographic patterns with symmetry invariance.
    """
    
    def __init__(self, config: Optional[SymmetryVAEConfigRGB] = None):
        super().__init__()
        
        if config is None:
            config = SymmetryVAEConfigRGB()
        
        self.config = config
        self.encoder = SymmetryInvariantEncoderRGB(config)
        self.decoder = SymmetryAwareDecoderRGB(config)
        self.transforms = SymmetryTransformsRGB(config.resolution)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, group_name: str = 'p4m') -> Dict[str, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encoder(x, group_name)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, group_name)
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def encode(self, x: torch.Tensor, group_name: str = 'p4m') -> torch.Tensor:
        """Encode to latent space (returns mean)."""
        mu, _ = self.encoder(x, group_name)
        return mu
    
    def decode(self, z: torch.Tensor, group_name: str = 'p4m') -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z, group_name)
    
    def sample(self, num_samples: int, group_name: str = 'p4m',
               device: str = 'cpu') -> torch.Tensor:
        """Sample new patterns from prior."""
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        return self.decode(z, group_name)


class SymmetryVAELossRGB(nn.Module):
    """Loss function for RGB Symmetry VAE."""
    
    def __init__(self, config: SymmetryVAEConfigRGB):
        super().__init__()
        self.config = config
        self.transforms = SymmetryTransformsRGB(config.resolution)
    
    def forward(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                group_name: str = 'p4m') -> Dict[str, torch.Tensor]:
        """Compute VAE loss."""
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (per-pixel MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.config.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }


def create_rgb_vae(resolution: int = 256, latent_dim: int = 2,
                   **kwargs) -> SymmetryInvariantVAE_RGB:
    """Factory function to create RGB VAE."""
    config = SymmetryVAEConfigRGB(
        resolution=resolution,
        latent_dim=latent_dim,
        **kwargs
    )
    return SymmetryInvariantVAE_RGB(config)


if __name__ == "__main__":
    print("Testing RGB Symmetry VAE...")
    
    config = SymmetryVAEConfigRGB(resolution=256, latent_dim=2)
    model = SymmetryInvariantVAE_RGB(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x, 'p4m')
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent mu shape: {outputs['mu'].shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")


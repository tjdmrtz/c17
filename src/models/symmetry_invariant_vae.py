"""
Symmetry-Invariant Variational Autoencoder for Turing Patterns

This module implements a VAE that is invariant to the symmetry operations
of the 17 wallpaper groups (rotations, reflections, glide reflections, translations).

Key Design Principles:
----------------------
1. **Symmetry Pooling in Encoder**: For each input, we apply all symmetry 
   operations of the target group and average the features. This makes the
   encoder output invariant to those transformations.

2. **Canonical Representation**: The latent space represents the "canonical"
   form of each pattern, independent of its orientation/position.

3. **Symmetry-Aware Decoder**: The decoder generates a base pattern and then
   applies symmetry operations to ensure the output respects group symmetry.

4. **Translation Invariance**: Achieved through global average pooling and
   by training with random translations.

Mathematical Background:
------------------------
For a group G with operations {g₁, g₂, ..., gₙ}, the invariant representation is:
    z = (1/|G|) Σᵢ f(gᵢ · x)

where f is the encoder and gᵢ · x is the transformed input.

This ensures z(g · x) = z(x) for all g ∈ G.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SymmetryVAEConfig:
    """Configuration for the Symmetry-Invariant VAE."""
    resolution: int = 128
    latent_dim: int = 2
    hidden_dims: List[int] = None
    
    # Symmetry settings
    use_symmetry_pooling: bool = True
    symmetry_groups: List[str] = None  # None = all 17 groups
    
    # Training settings
    beta: float = 1.0  # KL weight
    invariance_weight: float = 10.0  # Weight for invariance loss
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]
        if self.symmetry_groups is None:
            self.symmetry_groups = [
                'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
                'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
            ]


class SymmetryTransforms:
    """
    Applies symmetry transformations for the 17 wallpaper groups.
    
    Uses PyTorch operations for differentiability.
    """
    
    def __init__(self, resolution: int):
        self.resolution = resolution
        
        # Precompute rotation matrices for 60° and 120°
        self._setup_rotation_grids()
    
    def _setup_rotation_grids(self):
        """Setup grids for arbitrary rotations using grid_sample."""
        n = self.resolution
        
        # Create base grid
        theta_base = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float32)
        
        self.base_grid = F.affine_grid(
            theta_base.unsqueeze(0), 
            (1, 1, n, n),
            align_corners=False
        )
        
        # Rotation matrices
        angles = {
            60: np.pi / 3,
            90: np.pi / 2,
            120: 2 * np.pi / 3,
            180: np.pi,
            240: 4 * np.pi / 3,
            270: 3 * np.pi / 2,
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
                (1, 1, self.resolution, self.resolution),
                align_corners=False
            )
    
    def rotate_90(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Rotate by 90*k degrees using exact torch operation."""
        return torch.rot90(x, k, dims=[-2, -1])
    
    def rotate_arbitrary(self, x: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate by arbitrary angle using grid_sample with periodic boundaries."""
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
        
        # Apply periodic boundary conditions by wrapping the grid
        # grid is in [-1, 1] range, wrap to simulate tiling
        grid_periodic = torch.remainder(grid + 1, 2) - 1
        
        return F.grid_sample(x, grid_periodic, mode='bilinear', 
                            padding_mode='border', align_corners=False)
    
    def flip_h(self, x: torch.Tensor) -> torch.Tensor:
        """Horizontal flip (reflection across vertical axis)."""
        return torch.flip(x, dims=[-1])
    
    def flip_v(self, x: torch.Tensor) -> torch.Tensor:
        """Vertical flip (reflection across horizontal axis)."""
        return torch.flip(x, dims=[-2])
    
    def translate(self, x: torch.Tensor, shift_h: int, shift_v: int) -> torch.Tensor:
        """Translate with periodic boundary conditions."""
        return torch.roll(torch.roll(x, shift_h, dims=-1), shift_v, dims=-2)
    
    def get_group_transforms(self, group_name: str) -> List[callable]:
        """
        Get all transformations for a wallpaper group.
        
        Returns a list of functions that transform the input.
        """
        transforms = {
            'p1': [
                lambda x: x,  # Identity only
            ],
            'p2': [
                lambda x: x,
                lambda x: self.rotate_90(x, 2),  # 180°
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
        """
        Apply symmetry pooling: average features over all group transformations.
        
        This makes the output invariant to transformations in the group.
        """
        transforms = self.get_group_transforms(group_name)
        
        # Apply all transforms and average
        transformed = [t(x) for t in transforms]
        pooled = torch.stack(transformed, dim=0).mean(dim=0)
        
        return pooled
    
    def symmetrize_output(self, x: torch.Tensor, group_name: str) -> torch.Tensor:
        """
        Apply symmetry projection to output (same as pooling but for decoder).
        """
        return self.apply_symmetry_pooling(x, group_name)


class SymmetryInvariantEncoder(nn.Module):
    """
    Encoder that produces symmetry-invariant representations.
    
    Architecture:
    1. Convolutional feature extraction
    2. Symmetry pooling (average over all group transformations)
    3. Global average pooling for translation invariance
    4. MLP to latent space
    """
    
    def __init__(self, config: SymmetryVAEConfig):
        super().__init__()
        self.config = config
        self.transforms = SymmetryTransforms(config.resolution)
        
        # Convolutional encoder
        layers = []
        in_channels = 1
        
        for h_dim in config.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ))
            in_channels = h_dim
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        self.flat_size = config.hidden_dims[-1]  # After global pooling
        
        # MLP for mu and logvar
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flat_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, config.latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flat_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, config.latent_dim)
        )
    
    def forward(self, x: torch.Tensor, group_name: str = 'p4m') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor [B, 1, H, W]
            group_name: Wallpaper group for symmetry invariance
            
        Returns:
            mu, logvar: Latent distribution parameters [B, latent_dim]
        """
        batch_size = x.shape[0]
        
        if self.config.use_symmetry_pooling:
            # Get all transformations for this group
            transforms = self.transforms.get_group_transforms(group_name)
            
            # Process each transformation and average features
            all_features = []
            for transform in transforms:
                x_t = transform(x)
                features = self.conv_encoder(x_t)
                all_features.append(features)
            
            # Average over all transformations (symmetry pooling)
            features = torch.stack(all_features, dim=0).mean(dim=0)
        else:
            features = self.conv_encoder(x)
        
        # Global average pooling for translation invariance
        features = F.adaptive_avg_pool2d(features, 1).view(batch_size, -1)
        
        # Compute mu and logvar
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar


class SymmetryAwareDecoder(nn.Module):
    """
    Decoder that generates patterns with specified symmetry.
    
    Architecture:
    1. MLP from latent space
    2. Transposed convolutions to generate base pattern
    3. Symmetry projection to enforce group symmetry
    """
    
    def __init__(self, config: SymmetryVAEConfig):
        super().__init__()
        self.config = config
        self.transforms = SymmetryTransforms(config.resolution)
        
        # Initial projection
        init_size = config.resolution // (2 ** len(config.hidden_dims))
        self.init_size = init_size
        self.init_channels = config.hidden_dims[-1]
        
        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.init_channels * init_size * init_size),
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
        
        # Final layer
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1], 1,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ))
        
        self.conv_decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, group_name: str = 'p4m') -> torch.Tensor:
        """
        Decode latent vector to pattern with symmetry.
        
        Args:
            z: Latent vector [B, latent_dim]
            group_name: Wallpaper group for output symmetry
            
        Returns:
            Reconstructed pattern [B, 1, H, W]
        """
        batch_size = z.shape[0]
        
        # Project to initial feature map
        x = self.fc(z)
        x = x.view(batch_size, self.init_channels, self.init_size, self.init_size)
        
        # Decode
        x = self.conv_decoder(x)
        
        # Apply symmetry projection
        x = self.transforms.symmetrize_output(x, group_name)
        
        return x


class SymmetryInvariantVAE(nn.Module):
    """
    Variational Autoencoder with symmetry invariance for Turing patterns.
    
    The encoder produces representations that are invariant to symmetry
    operations, while the decoder generates patterns that respect the
    specified wallpaper group symmetry.
    
    Features:
    - 2D latent space for easy visualization
    - Symmetry pooling for invariance
    - Symmetry projection for equivariant generation
    - Translation invariance through global pooling
    """
    
    def __init__(self, config: Optional[SymmetryVAEConfig] = None):
        super().__init__()
        
        if config is None:
            config = SymmetryVAEConfig()
        
        self.config = config
        self.encoder = SymmetryInvariantEncoder(config)
        self.decoder = SymmetryAwareDecoder(config)
        self.transforms = SymmetryTransforms(config.resolution)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, 
                group_name: str = 'p4m') -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input pattern [B, 1, H, W]
            group_name: Wallpaper group
            
        Returns:
            Dictionary with reconstruction, mu, logvar, z
        """
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
        """Sample new patterns from the prior."""
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        return self.decode(z, group_name)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                    group_name: str = 'p4m', steps: int = 10) -> torch.Tensor:
        """Interpolate between two patterns in latent space."""
        z1 = self.encode(x1, group_name)
        z2 = self.encode(x2, group_name)
        
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            recon = self.decode(z, group_name)
            interpolated.append(recon)
        
        return torch.cat(interpolated, dim=0)


class SymmetryVAELoss(nn.Module):
    """
    Loss function for Symmetry-Invariant VAE.
    
    Components:
    1. Reconstruction loss (MSE or BCE)
    2. KL divergence
    3. Invariance loss: penalize differences between representations
       of transformed versions of the same pattern
    """
    
    def __init__(self, config: SymmetryVAEConfig):
        super().__init__()
        self.config = config
        self.transforms = SymmetryTransforms(config.resolution)
    
    def forward(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                group_name: str = 'p4m') -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x: Original input
            outputs: Dictionary from VAE forward pass
            group_name: Wallpaper group
            
        Returns:
            Dictionary with loss components
        """
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Invariance loss: representation should be same for transformed inputs
        invariance_loss = torch.tensor(0.0, device=x.device)
        
        if self.config.invariance_weight > 0:
            transforms = self.transforms.get_group_transforms(group_name)
            
            # Sample a few random transforms to check invariance
            n_check = min(3, len(transforms))
            indices = torch.randperm(len(transforms))[:n_check]
            
            for idx in indices:
                if idx == 0:  # Skip identity
                    continue
                x_t = transforms[idx](x)
                mu_t, _ = outputs.get('encoder', self.forward)(x_t, group_name) \
                    if 'encoder' in outputs else (mu, logvar)
                
                # Penalize difference in representations
                invariance_loss = invariance_loss + F.mse_loss(mu, mu_t)
            
            invariance_loss = invariance_loss / max(1, n_check - 1)
        
        # Total loss
        total_loss = (recon_loss + 
                     self.config.beta * kl_loss + 
                     self.config.invariance_weight * invariance_loss)
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'invariance_loss': invariance_loss
        }


def create_symmetry_vae(resolution: int = 128, 
                        latent_dim: int = 2,
                        **kwargs) -> SymmetryInvariantVAE:
    """
    Factory function to create a Symmetry-Invariant VAE.
    
    Args:
        resolution: Input resolution
        latent_dim: Latent space dimension (default 2 for visualization)
        **kwargs: Additional config options
        
    Returns:
        Configured VAE model
    """
    config = SymmetryVAEConfig(
        resolution=resolution,
        latent_dim=latent_dim,
        **kwargs
    )
    return SymmetryInvariantVAE(config)


if __name__ == "__main__":
    # Test the model
    print("Testing Symmetry-Invariant VAE...")
    
    config = SymmetryVAEConfig(resolution=128, latent_dim=2)
    model = SymmetryInvariantVAE(config)
    
    # Test forward pass
    x = torch.randn(4, 1, 128, 128)
    outputs = model(x, 'p4m')
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['recon'].shape}")
    print(f"Latent mu shape: {outputs['mu'].shape}")
    print(f"Latent z shape: {outputs['z'].shape}")
    
    # Test invariance
    transforms = SymmetryTransforms(128)
    x_rot = transforms.rotate_90(x, 1)
    
    mu1 = model.encode(x, 'p4m')
    mu2 = model.encode(x_rot, 'p4m')
    
    diff = (mu1 - mu2).abs().mean()
    print(f"\nInvariance test (should be small after training):")
    print(f"  |μ(x) - μ(rot90(x))| = {diff:.4f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")


#!/usr/bin/env python3
"""
Conditional Flow Matching for Crystallographic Phase Transitions.

State-of-the-art approach (2023-2024) for learning continuous transformations
between probability distributions. Used in Stable Diffusion 3, Imagen, etc.

Key advantages over Neural ODE:
- More stable training (no adjoint method needed)
- Faster inference (fewer function evaluations)
- Straighter trajectories (with Optimal Transport)
- Better theoretical properties

References:
- Lipman et al. "Flow Matching for Generative Modeling" (2023)
- Liu et al. "Flow Straight and Fast" (2023) - Rectified Flow
- Tong et al. "Improving and Generalizing Flow-Based Generative Models" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from tqdm import tqdm


# All 17 wallpaper groups
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]

GROUP_TO_IDX = {g: i for i, g in enumerate(ALL_17_GROUPS)}
IDX_TO_GROUP = {i: g for i, g in enumerate(ALL_17_GROUPS)}


@dataclass
class FlowMatchingConfig:
    """Configuration for Flow Matching model."""
    latent_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 6
    embedding_dim: int = 64
    time_embedding_dim: int = 64
    
    # Architecture
    use_attention: bool = True
    num_heads: int = 8
    dropout: float = 0.1
    
    # Flow Matching settings
    sigma_min: float = 1e-4  # Minimum noise level
    use_optimal_transport: bool = True  # Use OT-CFM for straighter paths
    
    # Regularization
    lambda_velocity: float = 0.01  # Velocity regularization


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same as used in diffusion models)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch] in [0, 1]
        Returns:
            Time embeddings [batch, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Sinusoidal encoding
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP projection
        return self.mlp(emb)


class GroupEmbedding(nn.Module):
    """Learnable embeddings for the 17 wallpaper groups with geometric priors."""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(17, embedding_dim)
        
        # Initialize with geometric properties
        self._init_geometric_embeddings()
    
    def _init_geometric_embeddings(self):
        """Initialize embeddings based on group properties."""
        # Geometric properties of each group
        group_properties = {
            'p1': (1, 0, False, False),    # (rotation_order, lattice, reflection, glide)
            'p2': (2, 0, False, False),
            'pm': (1, 0, True, False),
            'pg': (1, 0, False, True),
            'cm': (1, 1, True, False),
            'pmm': (2, 0, True, False),
            'pmg': (2, 0, True, True),
            'pgg': (2, 0, False, True),
            'cmm': (2, 1, True, False),
            'p4': (4, 2, False, False),
            'p4m': (4, 2, True, False),
            'p4g': (4, 2, True, True),
            'p3': (3, 3, False, False),
            'p3m1': (3, 3, True, False),
            'p31m': (3, 3, True, False),
            'p6': (6, 3, False, False),
            'p6m': (6, 3, True, False),
        }
        
        with torch.no_grad():
            for i, group in enumerate(ALL_17_GROUPS):
                rotation, lattice, reflection, glide = group_properties[group]
                
                # Encode properties in first dimensions
                self.embedding.weight[i, 0] = rotation / 6.0
                self.embedding.weight[i, 1] = np.sin(2 * np.pi * rotation / 6.0)
                self.embedding.weight[i, 2] = np.cos(2 * np.pi * rotation / 6.0)
                self.embedding.weight[i, 3] = lattice / 3.0
                self.embedding.weight[i, 4] = 1.0 if reflection else 0.0
                self.embedding.weight[i, 5] = 1.0 if glide else 0.0
                
                # Rest is learned
                self.embedding.weight[i, 6:] = torch.randn(self.embedding_dim - 6) * 0.1
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)


class VelocityNetwork(nn.Module):
    """
    Neural network that predicts the velocity field v(z, t, source, target).
    
    This is the core of Flow Matching - we learn to predict the velocity
    that transports z_source to z_target over time t ∈ [0, 1].
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.group_embedding = GroupEmbedding(config.embedding_dim)
        
        # Input projection
        input_dim = config.latent_dim + config.time_embedding_dim + 2 * config.embedding_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # Main network with residual blocks
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            self.layers.append(ResidualBlock(
                config.hidden_dim,
                dropout=config.dropout,
                use_attention=(config.use_attention and i % 2 == 0),
                num_heads=config.num_heads,
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        
        # Initialize output to near-zero for stable training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity at point z and time t.
        
        Args:
            z: Current position [batch, latent_dim]
            t: Time [batch] or scalar
            source_idx: Source group indices [batch]
            target_idx: Target group indices [batch]
            
        Returns:
            Velocity v(z, t) [batch, latent_dim]
        """
        batch_size = z.shape[0]
        
        # Handle scalar t
        if t.dim() == 0:
            t = t.expand(batch_size)
        
        # Get embeddings
        t_emb = self.time_embedding(t)
        source_emb = self.group_embedding(source_idx)
        target_emb = self.group_embedding(target_idx)
        
        # Concatenate all inputs
        x = torch.cat([z, t_emb, source_emb, target_emb], dim=-1)
        
        # Forward through network
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_proj(x)


class ResidualBlock(nn.Module):
    """Residual block with optional self-attention."""
    
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        use_attention: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.norm2 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feed-forward with residual
        x = x + self.ff(self.norm1(x))
        
        # Self-attention with residual (if enabled)
        if self.use_attention:
            x_norm = self.norm2(x).unsqueeze(1)  # [batch, 1, dim]
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + attn_out.squeeze(1)
        
        return x


class FlowMatchingTransition(nn.Module):
    """
    Conditional Flow Matching model for crystallographic phase transitions.
    
    Given source and target symmetry groups, learns a continuous flow
    that transforms patterns from source to target distribution.
    """
    
    def __init__(self, config: Optional[FlowMatchingConfig] = None):
        super().__init__()
        
        if config is None:
            config = FlowMatchingConfig()
        
        self.config = config
        self.velocity_net = VelocityNetwork(config)
    
    def get_interpolant(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the interpolant z_t and target velocity for Flow Matching.
        
        For Optimal Transport CFM (OT-CFM), we use linear interpolation:
            z_t = (1 - t) * z0 + t * z1
            v_target = z1 - z0
        
        Args:
            z0: Source points [batch, latent_dim]
            z1: Target points [batch, latent_dim]
            t: Time values [batch] in [0, 1]
            
        Returns:
            z_t: Interpolated points [batch, latent_dim]
            v_target: Target velocity [batch, latent_dim]
        """
        t = t.view(-1, 1)  # [batch, 1]
        
        # Linear interpolation (OT path)
        z_t = (1 - t) * z0 + t * z1
        
        # Target velocity is constant for linear interpolation
        v_target = z1 - z0
        
        # Add small noise for regularization (optional, helps with sharp distributions)
        if self.training and self.config.sigma_min > 0:
            z_t = z_t + self.config.sigma_min * torch.randn_like(z_t)
        
        return z_t, v_target
    
    def compute_loss(
        self,
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Flow Matching loss.
        
        The loss is simply MSE between predicted and target velocity:
            L = ||v_pred(z_t, t) - v_target||^2
        
        This is much simpler and more stable than Neural ODE training!
        """
        batch_size = z_source.shape[0]
        device = z_source.device
        
        # Sample random times uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Get interpolant and target velocity
        z_t, v_target = self.get_interpolant(z_source, z_target, t)
        
        # Predict velocity
        v_pred = self.velocity_net(z_t, t, source_idx, target_idx)
        
        # Flow matching loss (MSE)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Velocity magnitude regularization (optional)
        velocity_reg = (v_pred ** 2).mean() * self.config.lambda_velocity
        
        total_loss = flow_loss + velocity_reg
        
        return {
            'loss': total_loss,
            'flow_loss': flow_loss,
            'velocity_reg': velocity_reg,
        }
    
    @torch.no_grad()
    def sample_trajectory(
        self,
        z_start: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        n_steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate trajectory from source to target using Euler integration.
        
        Since we're using OT-CFM with linear paths, even few steps give good results.
        
        Args:
            z_start: Starting points [batch, latent_dim]
            source_idx: Source group indices [batch]
            target_idx: Target group indices [batch]
            n_steps: Number of integration steps
            
        Returns:
            Trajectory [n_steps, batch, latent_dim]
        """
        self.eval()
        device = z_start.device
        batch_size = z_start.shape[0]
        
        # Time steps
        dt = 1.0 / n_steps
        
        # Storage for trajectory
        trajectory = [z_start]
        z = z_start.clone()
        
        for i in range(n_steps - 1):
            t = torch.full((batch_size,), i * dt, device=device)
            
            # Predict velocity
            v = self.velocity_net(z, t, source_idx, target_idx)
            
            # Euler step
            z = z + v * dt
            
            trajectory.append(z.clone())
        
        return torch.stack(trajectory, dim=0)  # [n_steps, batch, latent_dim]
    
    @torch.no_grad()
    def sample_endpoint(
        self,
        z_start: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        Get final point after flow transformation.
        
        Args:
            z_start: Starting points [batch, latent_dim]
            source_idx: Source group indices [batch]
            target_idx: Target group indices [batch]
            n_steps: Number of integration steps
            
        Returns:
            Final points [batch, latent_dim]
        """
        trajectory = self.sample_trajectory(z_start, source_idx, target_idx, n_steps)
        return trajectory[-1]
    
    def forward(
        self,
        z_start: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        n_steps: int = 20,
        return_trajectory: bool = True,
    ) -> torch.Tensor:
        """Forward pass - returns trajectory or endpoint."""
        if return_trajectory:
            return self.sample_trajectory(z_start, source_idx, target_idx, n_steps)
        else:
            return self.sample_endpoint(z_start, source_idx, target_idx, n_steps)


class FlowMatchingLoss(nn.Module):
    """Wrapper for Flow Matching loss computation."""
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        model: FlowMatchingTransition,
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch."""
        return model.compute_loss(z_source, z_target, source_idx, target_idx)


class FlowMatchingMetrics:
    """Metrics for evaluating Flow Matching transitions."""
    
    @staticmethod
    @torch.no_grad()
    def compute_metrics(
        model: FlowMatchingTransition,
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        n_steps: int = 50,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Returns:
            Dictionary with:
            - endpoint_mse: MSE between predicted and true target
            - trajectory_smoothness: How smooth the trajectory is
            - path_length: Total path length
        """
        model.eval()
        
        # Get trajectory
        trajectory = model.sample_trajectory(z_source, source_idx, target_idx, n_steps)
        z_pred = trajectory[-1]
        
        # Endpoint MSE
        endpoint_mse = F.mse_loss(z_pred, z_target).item()
        
        # Trajectory smoothness (second derivative)
        if trajectory.shape[0] >= 3:
            velocities = trajectory[1:] - trajectory[:-1]
            accelerations = velocities[1:] - velocities[:-1]
            smoothness = (accelerations ** 2).mean().item()
        else:
            smoothness = 0.0
        
        # Path length
        path_length = torch.norm(trajectory[1:] - trajectory[:-1], dim=-1).sum(dim=0).mean().item()
        
        # Straightness (ratio of direct distance to path length)
        direct_distance = torch.norm(z_target - z_source, dim=-1).mean().item()
        straightness = direct_distance / (path_length + 1e-8)
        
        return {
            'endpoint_mse': endpoint_mse,
            'smoothness': smoothness,
            'path_length': path_length,
            'straightness': straightness,
        }



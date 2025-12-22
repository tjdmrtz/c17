"""
Neural ODE for Crystallographic Phase Transitions.

This module implements the Neural ODE that learns to simulate
continuous phase transitions between the 17 wallpaper groups.

Mathematical formulation:
    dz/dt = f_θ(z, t, e_source, e_target)

where:
    - z: latent representation (2D)
    - t: transition time [0, 1]
    - e_source, e_target: group embeddings
    - f_θ: neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not installed. Install with: pip install torchdiffeq")


# All 17 wallpaper groups
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]

GROUP_TO_IDX = {g: i for i, g in enumerate(ALL_17_GROUPS)}
IDX_TO_GROUP = {i: g for i, g in enumerate(ALL_17_GROUPS)}

# Lattice info for structured embeddings
LATTICE_INFO = {
    'p1': ('Oblique', 1, False, False),
    'p2': ('Oblique', 2, False, False),
    'pm': ('Rectangular', 1, True, False),
    'pg': ('Rectangular', 1, False, True),
    'cm': ('Rectangular', 1, True, False),
    'pmm': ('Rectangular', 2, True, False),
    'pmg': ('Rectangular', 2, True, True),
    'pgg': ('Rectangular', 2, False, True),
    'cmm': ('Rectangular', 2, True, False),
    'p4': ('Square', 4, False, False),
    'p4m': ('Square', 4, True, False),
    'p4g': ('Square', 4, True, True),
    'p3': ('Hexagonal', 3, False, False),
    'p3m1': ('Hexagonal', 3, True, False),
    'p31m': ('Hexagonal', 3, True, False),
    'p6': ('Hexagonal', 6, False, False),
    'p6m': ('Hexagonal', 6, True, False),
}

LATTICE_TO_IDX = {'Oblique': 0, 'Rectangular': 1, 'Square': 2, 'Hexagonal': 3}


@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODE transition model."""
    latent_dim: int = 2
    hidden_dim: int = 256
    embedding_dim: int = 32
    num_layers: int = 4
    use_residual: bool = True
    
    # ODE solver settings
    solver: str = 'rk4'  # More stable than euler
    rtol: float = 1e-3
    atol: float = 1e-4
    use_adjoint: bool = True
    
    # Loss weights
    lambda_smooth: float = 0.1
    lambda_velocity: float = 0.01


class GroupEmbedding(nn.Module):
    """
    Learnable embeddings for the 17 wallpaper groups.
    
    Initialized with structured information about group properties.
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(17, embedding_dim)
        
        self._init_structured_embeddings()
    
    def _init_structured_embeddings(self):
        """Initialize embeddings with group structure."""
        with torch.no_grad():
            for i, group in enumerate(ALL_17_GROUPS):
                lattice, rotation, has_reflection, has_glide = LATTICE_INFO[group]
                
                # Dims 0-3: Lattice type one-hot
                self.embedding.weight[i, :4] = 0
                self.embedding.weight[i, LATTICE_TO_IDX[lattice]] = 1.0
                
                # Dims 4-5: Rotation order (normalized)
                self.embedding.weight[i, 4] = rotation / 6.0
                self.embedding.weight[i, 5] = np.sin(2 * np.pi * rotation / 6.0)
                
                # Dims 6-7: Reflection and glide flags
                self.embedding.weight[i, 6] = 1.0 if has_reflection else 0.0
                self.embedding.weight[i, 7] = 1.0 if has_glide else 0.0
                
                # Dims 8-31: Random initialization (will be learned)
                self.embedding.weight[i, 8:] = torch.randn(self.embedding_dim - 8) * 0.1
    
    def forward(self, group_idx: torch.Tensor) -> torch.Tensor:
        """Get embeddings for group indices."""
        return self.embedding(group_idx)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all 17 group embeddings."""
        indices = torch.arange(17, device=self.embedding.weight.device)
        return self.embedding(indices)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create time embedding.
        
        Args:
            t: Time value (scalar or batch)
            
        Returns:
            Time embedding of shape (batch, dim) or (dim,)
        """
        # Sinusoidal encoding
        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return self.mlp(embedding)


class ODEFunc(nn.Module):
    """
    Neural network that defines dz/dt = f(z, t, e_source, e_target).
    
    This is the core dynamics function of the Neural ODE.
    """
    
    def __init__(self, config: NeuralODEConfig):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Embeddings
        self.group_embedding = GroupEmbedding(config.embedding_dim)
        self.time_embedding = TimeEmbedding(32)
        
        # Input: z + time_embed + source_embed + target_embed
        input_dim = config.latent_dim + 32 + 2 * config.embedding_dim
        
        # Build MLP with optional residual connections
        layers = []
        current_dim = input_dim
        
        for i in range(config.num_layers):
            if i == 0:
                layers.append(nn.Linear(current_dim, config.hidden_dim))
                current_dim = config.hidden_dim
            else:
                layers.append(nn.Linear(current_dim, config.hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(config.hidden_dim, config.latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Residual projection if dimensions don't match
        if config.use_residual:
            self.res_proj = nn.Linear(input_dim, config.latent_dim)
        
        # Store conditioning (set before ODE solve)
        self.source_idx = None
        self.target_idx = None
        self._source_embed = None
        self._target_embed = None
    
    def set_condition(self, source_idx: torch.Tensor, target_idx: torch.Tensor):
        """Set source and target groups for current forward pass."""
        self.source_idx = source_idx
        self.target_idx = target_idx
        
        # Precompute embeddings
        self._source_embed = self.group_embedding(source_idx)
        self._target_embed = self.group_embedding(target_idx)
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute dz/dt at time t.
        
        Args:
            t: Current time (scalar tensor)
            z: Current latent state [batch, latent_dim]
            
        Returns:
            dz/dt: Rate of change [batch, latent_dim]
        """
        batch_size = z.shape[0]
        
        # Time embedding
        t_embed = self.time_embedding(t)
        if t_embed.shape[0] == 1:
            t_embed = t_embed.expand(batch_size, -1)
        
        # Group embeddings (expand if needed)
        source_embed = self._source_embed
        target_embed = self._target_embed
        
        if source_embed.shape[0] == 1:
            source_embed = source_embed.expand(batch_size, -1)
            target_embed = target_embed.expand(batch_size, -1)
        
        # Concatenate all inputs
        x = torch.cat([z, t_embed, source_embed, target_embed], dim=-1)
        
        # Forward through network
        dz_dt = self.net(x)
        
        # Optional residual
        if self.config.use_residual:
            dz_dt = dz_dt + 0.1 * self.res_proj(x)
        
        return dz_dt


class NeuralODETransition(nn.Module):
    """
    Neural ODE model for crystallographic phase transitions.
    
    Given a latent point with source symmetry, evolves it
    to target symmetry over time t ∈ [0, 1].
    """
    
    def __init__(self, config: Optional[NeuralODEConfig] = None):
        super().__init__()
        
        if config is None:
            config = NeuralODEConfig()
        
        self.config = config
        self.ode_func = ODEFunc(config)
        
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq is required. Install with: pip install torchdiffeq")
    
    def forward(
        self,
        z_start: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        n_steps: int = 50,
        return_trajectory: bool = True,
    ) -> torch.Tensor:
        """
        Evolve latent state from source to target symmetry.
        
        Args:
            z_start: Initial latent [batch, latent_dim]
            source_idx: Source group indices [batch] or [1]
            target_idx: Target group indices [batch] or [1]
            n_steps: Number of output timesteps
            return_trajectory: If True, return full trajectory
            
        Returns:
            If return_trajectory: [n_steps, batch, latent_dim]
            Else: [batch, latent_dim] (final state only)
        """
        device = z_start.device
        
        # Set conditioning
        self.ode_func.set_condition(source_idx, target_idx)
        
        # Time points
        t = torch.linspace(0, 1, n_steps, device=device)
        
        # Choose solver
        if self.config.use_adjoint:
            solver = odeint_adjoint
        else:
            solver = odeint
        
        # Solve ODE
        trajectory = solver(
            self.ode_func,
            z_start,
            t,
            method=self.config.solver,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )
        
        if return_trajectory:
            return trajectory  # [n_steps, batch, latent_dim]
        else:
            return trajectory[-1]  # [batch, latent_dim]
    
    def forward_by_name(
        self,
        z_start: torch.Tensor,
        source_group: str,
        target_group: str,
        n_steps: int = 50,
    ) -> torch.Tensor:
        """Forward using group names instead of indices."""
        device = z_start.device
        source_idx = torch.tensor([GROUP_TO_IDX[source_group]], device=device)
        target_idx = torch.tensor([GROUP_TO_IDX[target_group]], device=device)
        
        return self.forward(z_start, source_idx, target_idx, n_steps)


class TransitionLoss(nn.Module):
    """
    Loss function for training Neural ODE transitions.
    
    Components:
    1. Endpoint loss: MSE between predicted and target latent
    2. Smoothness loss: Penalize non-smooth trajectories
    3. Velocity loss: Regularize trajectory speed
    """
    
    def __init__(self, config: NeuralODEConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        trajectory: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute transition loss.
        
        Args:
            trajectory: [n_steps, batch, latent_dim]
            z_target: Target latent [batch, latent_dim]
            
        Returns:
            Dict with loss components
        """
        n_steps = trajectory.shape[0]
        
        # Endpoint loss
        z_pred = trajectory[-1]
        endpoint_loss = F.mse_loss(z_pred, z_target)
        
        # Smoothness loss: penalize acceleration (second derivative)
        if n_steps >= 3:
            velocity = trajectory[1:] - trajectory[:-1]  # First derivative
            acceleration = velocity[1:] - velocity[:-1]  # Second derivative
            smoothness_loss = (acceleration ** 2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=trajectory.device)
        
        # Velocity regularization: prefer constant speed
        velocity = trajectory[1:] - trajectory[:-1]
        velocity_norms = torch.norm(velocity, dim=-1)
        velocity_loss = velocity_norms.var(dim=0).mean()  # Variance of speed
        
        # Total loss
        total_loss = (
            endpoint_loss +
            self.config.lambda_smooth * smoothness_loss +
            self.config.lambda_velocity * velocity_loss
        )
        
        return {
            'loss': total_loss,
            'endpoint_loss': endpoint_loss,
            'smoothness_loss': smoothness_loss,
            'velocity_loss': velocity_loss,
        }


def compute_nfe(trajectory: torch.Tensor) -> int:
    """Estimate number of function evaluations from trajectory."""
    # This is a rough estimate - actual NFE depends on adaptive solver
    return trajectory.shape[0] * 2  # Approximate for dopri5


def compute_trajectory_length(trajectory: torch.Tensor) -> torch.Tensor:
    """Compute arc length of trajectory."""
    # trajectory: [n_steps, batch, latent_dim]
    diffs = trajectory[1:] - trajectory[:-1]
    lengths = torch.norm(diffs, dim=-1)
    return lengths.sum(dim=0)  # [batch]


class TransitionMetrics:
    """Utility class for computing transition metrics."""
    
    @staticmethod
    def endpoint_error(trajectory: torch.Tensor, z_target: torch.Tensor) -> float:
        """Mean endpoint error."""
        z_pred = trajectory[-1]
        return F.mse_loss(z_pred, z_target).item()
    
    @staticmethod
    def trajectory_smoothness(trajectory: torch.Tensor) -> float:
        """Smoothness score (lower = smoother)."""
        if trajectory.shape[0] < 3:
            return 0.0
        velocity = trajectory[1:] - trajectory[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        return (acceleration ** 2).mean().item()
    
    @staticmethod
    def mean_trajectory_length(trajectory: torch.Tensor) -> float:
        """Mean arc length of trajectories."""
        lengths = compute_trajectory_length(trajectory)
        return lengths.mean().item()


if __name__ == "__main__":
    print("Testing Neural ODE Transition...")
    
    config = NeuralODEConfig(latent_dim=2, hidden_dim=128)
    model = NeuralODETransition(config)
    
    # Test forward pass
    batch_size = 4
    z_start = torch.randn(batch_size, 2)
    source_idx = torch.zeros(batch_size, dtype=torch.long)  # p1
    target_idx = torch.full((batch_size,), 16, dtype=torch.long)  # p6m
    
    trajectory = model(z_start, source_idx, target_idx, n_steps=10)
    
    print(f"Input shape: {z_start.shape}")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Start: {trajectory[0, 0].detach().numpy()}")
    print(f"End: {trajectory[-1, 0].detach().numpy()}")
    
    # Test loss
    z_target = torch.randn(batch_size, 2)
    loss_fn = TransitionLoss(config)
    losses = loss_fn(trajectory, z_target)
    
    print(f"\nLosses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")



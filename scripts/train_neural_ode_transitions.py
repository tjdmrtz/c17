#!/usr/bin/env python3
"""
Neural ODE for Crystallographic Phase Transitions

This script implements a Neural ODE that learns to simulate continuous
phase transitions between the 17 wallpaper symmetry groups.

The model learns the dynamics of how patterns transform from one
crystallographic symmetry to another, enabling:
- Smooth interpolation between symmetry groups
- Visualization of phase transition dynamics
- Physical understanding of symmetry breaking/formation

Mathematical Framework:
----------------------
dz/dt = f_θ(z, t, g_source, g_target)

where:
- z: latent representation of the pattern
- t: time parameter (0 = source symmetry, 1 = target symmetry)
- g_source, g_target: source and target symmetry groups
- f_θ: neural network parameterizing the dynamics

Usage:
    python scripts/train_neural_ode_transitions.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from torchdiffeq import odeint, odeint_adjoint

from src.models.symmetry_invariant_vae import (
    SymmetryInvariantVAE,
    SymmetryVAEConfig
)


# All 17 wallpaper groups
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]

# Group to index mapping
GROUP_TO_IDX = {g: i for i, g in enumerate(ALL_17_GROUPS)}
IDX_TO_GROUP = {i: g for i, g in enumerate(ALL_17_GROUPS)}

# Lattice type info for visualization
LATTICE_INFO = {
    'p1': ('Oblique', '#1a5276'), 'p2': ('Oblique', '#2980b9'),
    'pm': ('Rectangular', '#27ae60'), 'pg': ('Rectangular', '#2ecc71'),
    'cm': ('Rectangular', '#58d68d'), 'pmm': ('Rectangular', '#145a32'),
    'pmg': ('Rectangular', '#196f3d'), 'pgg': ('Rectangular', '#1d8348'),
    'cmm': ('Rectangular', '#239b56'),
    'p4': ('Square', '#6c3483'), 'p4m': ('Square', '#8e44ad'),
    'p4g': ('Square', '#a569bd'),
    'p3': ('Hexagonal', '#b9770e'), 'p3m1': ('Hexagonal', '#d68910'),
    'p31m': ('Hexagonal', '#f39c12'), 'p6': ('Hexagonal', '#c0392b'),
    'p6m': ('Hexagonal', '#e74c3c'),
}


class GroupEmbedding(nn.Module):
    """Learnable embeddings for the 17 wallpaper groups."""
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(17, embedding_dim)
        
        # Initialize with structure-aware embeddings
        # Groups with similar symmetries should be closer
        self._init_structured_embeddings()
    
    def _init_structured_embeddings(self):
        """Initialize embeddings based on group relationships."""
        # Lattice type encoding (one-hot style initialization)
        lattice_encoding = torch.zeros(17, 4)
        for i, group in enumerate(ALL_17_GROUPS):
            lattice = LATTICE_INFO[group][0]
            if lattice == 'Oblique':
                lattice_encoding[i, 0] = 1
            elif lattice == 'Rectangular':
                lattice_encoding[i, 1] = 1
            elif lattice == 'Square':
                lattice_encoding[i, 2] = 1
            elif lattice == 'Hexagonal':
                lattice_encoding[i, 3] = 1
        
        # Use first 4 dims for lattice type, rest random
        with torch.no_grad():
            self.embedding.weight[:, :4] = lattice_encoding
    
    def forward(self, group_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(group_idx)


class ODEFunc(nn.Module):
    """
    Neural network that defines the ODE dynamics for phase transitions.
    
    dz/dt = f(z, t, g_source, g_target)
    
    The network learns how patterns should evolve to transition
    from one symmetry group to another.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128, 
                 group_embedding_dim: int = 32):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.group_embedding = GroupEmbedding(group_embedding_dim)
        
        # Input: z + t + source_embedding + target_embedding
        input_dim = latent_dim + 1 + 2 * group_embedding_dim
        
        # MLP for dynamics
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Learnable time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
        # Store conditioning for ODE solver
        self.source_idx = None
        self.target_idx = None
    
    def set_condition(self, source_idx: torch.Tensor, target_idx: torch.Tensor):
        """Set the source and target groups for the transition."""
        self.source_idx = source_idx
        self.target_idx = target_idx
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute dz/dt at time t.
        
        Args:
            t: Current time (scalar)
            z: Current latent state [batch, latent_dim]
            
        Returns:
            dz/dt: Rate of change [batch, latent_dim]
        """
        batch_size = z.shape[0]
        
        # Time embedding
        t_embed = self.time_embed(t.view(1, 1)).expand(batch_size, 1)
        
        # Group embeddings
        source_embed = self.group_embedding(self.source_idx)
        target_embed = self.group_embedding(self.target_idx)
        
        # Expand if needed
        if source_embed.shape[0] == 1:
            source_embed = source_embed.expand(batch_size, -1)
            target_embed = target_embed.expand(batch_size, -1)
        
        # Concatenate all inputs
        x = torch.cat([z, t_embed, source_embed, target_embed], dim=-1)
        
        # Compute dynamics
        dz_dt = self.net(x)
        
        return dz_dt


class NeuralODETransition(nn.Module):
    """
    Neural ODE model for crystallographic phase transitions.
    
    Given a pattern with source symmetry, this model can evolve it
    continuously to a target symmetry group.
    """
    
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 256,
                 use_adjoint: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.use_adjoint = use_adjoint
        
        # Integration method
        self.solver = 'dopri5'  # Adaptive Runge-Kutta
        self.rtol = 1e-3
        self.atol = 1e-4
    
    def forward(self, z_start: torch.Tensor, 
                source_group: str, target_group: str,
                n_steps: int = 50) -> torch.Tensor:
        """
        Evolve latent state from source to target symmetry.
        
        Args:
            z_start: Initial latent state [batch, latent_dim]
            source_group: Source symmetry group name
            target_group: Target symmetry group name
            n_steps: Number of output time steps
            
        Returns:
            Trajectory of latent states [n_steps, batch, latent_dim]
        """
        device = z_start.device
        batch_size = z_start.shape[0]
        
        # Get group indices
        source_idx = torch.tensor([GROUP_TO_IDX[source_group]], device=device)
        target_idx = torch.tensor([GROUP_TO_IDX[target_group]], device=device)
        
        # Set conditioning
        self.ode_func.set_condition(source_idx, target_idx)
        
        # Time points from 0 to 1
        t = torch.linspace(0, 1, n_steps, device=device)
        
        # Solve ODE
        ode_solver = odeint_adjoint if self.use_adjoint else odeint
        
        trajectory = ode_solver(
            self.ode_func, z_start, t,
            method=self.solver,
            rtol=self.rtol, atol=self.atol
        )
        
        return trajectory  # [n_steps, batch, latent_dim]
    
    def compute_endpoint(self, z_start: torch.Tensor,
                         source_group: str, target_group: str) -> torch.Tensor:
        """Compute only the final state (more efficient for training)."""
        trajectory = self.forward(z_start, source_group, target_group, n_steps=2)
        return trajectory[-1]  # [batch, latent_dim]


class TransitionDataset(Dataset):
    """
    Dataset of (source_pattern, source_group, target_group, target_pattern) tuples.
    
    Uses a pretrained VAE to encode patterns into latent space.
    """
    
    def __init__(self, vae: SymmetryInvariantVAE, device: torch.device,
                 num_pairs: int = 5000, seed: int = 42):
        super().__init__()
        
        self.vae = vae
        self.device = device
        self.num_pairs = num_pairs
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.pairs = []
        self._generate_pairs()
    
    def _generate_pairs(self):
        """Generate transition pairs by sampling from latent space."""
        print("Generating transition pairs...")
        
        self.vae.eval()
        
        with torch.no_grad():
            for _ in tqdm(range(self.num_pairs)):
                # Random source and target groups
                source_idx = np.random.randint(0, 17)
                target_idx = np.random.randint(0, 17)
                
                source_group = ALL_17_GROUPS[source_idx]
                target_group = ALL_17_GROUPS[target_idx]
                
                # Sample a random latent point
                z = torch.randn(1, self.vae.config.latent_dim, device=self.device)
                
                # Decode with source symmetry, then encode to get "canonical" latent
                pattern_source = self.vae.decode(z, source_group)
                z_source = self.vae.encode(pattern_source, source_group)
                
                # Decode with target symmetry to get target latent
                pattern_target = self.vae.decode(z, target_group)
                z_target = self.vae.encode(pattern_target, target_group)
                
                self.pairs.append({
                    'z_source': z_source.cpu(),
                    'z_target': z_target.cpu(),
                    'source_group': source_group,
                    'target_group': target_group,
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return (
            pair['z_source'].squeeze(0),
            pair['z_target'].squeeze(0),
            pair['source_idx'],
            pair['target_idx'],
        )


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        z_source, z_target, source_idx, target_idx = batch
        z_source = z_source.to(device)
        z_target = z_target.to(device)
        
        # Get unique source/target pairs in batch
        # For simplicity, use first pair's groups for entire batch
        source_group = ALL_17_GROUPS[source_idx[0].item()]
        target_group = ALL_17_GROUPS[target_idx[0].item()]
        
        optimizer.zero_grad()
        
        # Compute endpoint prediction
        z_pred = model.compute_endpoint(z_source, source_group, target_group)
        
        # MSE loss between predicted and target latent
        loss = F.mse_loss(z_pred, z_target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def visualize_transition(model, vae, device, source_group, target_group,
                         save_path, n_frames=30, seed=42):
    """Visualize a phase transition as a sequence of images."""
    
    torch.manual_seed(seed)
    
    model.eval()
    vae.eval()
    
    # Sample starting point
    z_start = torch.randn(1, vae.config.latent_dim, device=device) * 0.8
    
    # Get trajectory
    with torch.no_grad():
        trajectory = model(z_start, source_group, target_group, n_steps=n_frames)
    
    # Decode each point using interpolated symmetry
    images = []
    
    with torch.no_grad():
        for i, z in enumerate(trajectory):
            # Use source symmetry at start, target at end
            t = i / (n_frames - 1)
            
            if t < 0.5:
                img = vae.decode(z, source_group)
            else:
                img = vae.decode(z, target_group)
            
            images.append(img[0, 0].cpu().numpy())
    
    # Create figure
    n_cols = min(10, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    fig.patch.set_facecolor('#0a0a0a')
    
    source_color = LATTICE_INFO[source_group][1]
    target_color = LATTICE_INFO[target_group][1]
    
    for i, ax in enumerate(axes.flat):
        if i < n_frames:
            ax.imshow(images[i], cmap='viridis')
            t = i / (n_frames - 1)
            
            # Color interpolation for border
            ax.set_facecolor('#0a0a0a')
            for spine in ax.spines.values():
                spine.set_visible(True)
                # Interpolate color
                if t < 0.5:
                    spine.set_color(source_color)
                else:
                    spine.set_color(target_color)
                spine.set_linewidth(2)
        ax.axis('off')
    
    fig.suptitle(f'Phase Transition: {source_group} → {target_group}',
                 fontsize=16, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved transition visualization to {save_path}")


def visualize_dual_transition(model, vae, device, source_group, target_group,
                               save_path, n_frames=12, seed=42):
    """
    Visualize transition showing BOTH latent space trajectory AND pattern evolution.
    
    Creates a figure with:
    - Left: Latent space with trajectory
    - Right: Sequence of decoded patterns
    """
    torch.manual_seed(seed)
    
    model.eval()
    vae.eval()
    
    # Sample starting point
    z_start = torch.randn(1, vae.config.latent_dim, device=device) * 0.8
    
    # Get trajectory
    with torch.no_grad():
        trajectory = model(z_start, source_group, target_group, n_steps=n_frames)
    
    # Collect latent points and images
    latent_points = []
    images = []
    
    with torch.no_grad():
        for i, z in enumerate(trajectory):
            latent_points.append(z[0].cpu().numpy())
            
            t = i / (n_frames - 1)
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            
            img_source = vae.decode(z, source_group)
            img_target = vae.decode(z, target_group)
            img = (1 - alpha) * img_source + alpha * img_target
            
            images.append(img[0, 0].cpu().numpy())
    
    latent_points = np.array(latent_points)
    
    # Create figure: latent space on left, patterns on right
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Left: Latent space trajectory
    ax_latent = fig.add_subplot(1, 2, 1)
    ax_latent.set_facecolor('#0a0a0a')
    
    source_color = LATTICE_INFO[source_group][1]
    target_color = LATTICE_INFO[target_group][1]
    
    # Plot trajectory
    colors = np.linspace(0, 1, n_frames)
    for i in range(n_frames - 1):
        ax_latent.plot(latent_points[i:i+2, 0], latent_points[i:i+2, 1],
                      color=plt.cm.coolwarm(colors[i]), linewidth=3, alpha=0.8)
    
    # Mark start and end
    ax_latent.scatter(latent_points[0, 0], latent_points[0, 1], 
                     c=source_color, s=200, marker='o', edgecolors='white',
                     linewidths=2, zorder=10, label=f'Start ({source_group})')
    ax_latent.scatter(latent_points[-1, 0], latent_points[-1, 1],
                     c=target_color, s=200, marker='*', edgecolors='white',
                     linewidths=2, zorder=10, label=f'End ({target_group})')
    
    # Mark intermediate points
    for i in range(1, n_frames - 1):
        t = i / (n_frames - 1)
        ax_latent.scatter(latent_points[i, 0], latent_points[i, 1],
                         c=[plt.cm.coolwarm(t)], s=50, alpha=0.7, edgecolors='white')
    
    ax_latent.set_xlabel('z₁', fontsize=14, color='white')
    ax_latent.set_ylabel('z₂', fontsize=14, color='white')
    ax_latent.set_title('Latent Space Trajectory', fontsize=16, color='white', fontweight='bold')
    ax_latent.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white',
                    labelcolor='white', fontsize=10)
    ax_latent.tick_params(colors='white')
    for spine in ax_latent.spines.values():
        spine.set_color('white')
    ax_latent.set_aspect('equal')
    ax_latent.grid(True, alpha=0.2, color='white')
    
    # Right: Pattern sequence
    n_cols = 4
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    gs = fig.add_gridspec(n_rows, n_cols, left=0.52, right=0.98, 
                          top=0.88, bottom=0.08, hspace=0.15, wspace=0.08)
    
    for i in range(n_frames):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0a0a')
        
        t = i / (n_frames - 1)
        ax.imshow(images[i], cmap='viridis')
        ax.set_title(f't={t:.2f}', fontsize=10, color='white')
        ax.axis('off')
        
        # Border color interpolation
        border_color = plt.cm.coolwarm(t)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(2)
    
    fig.suptitle(f'Phase Transition: {source_group} → {target_group}\n'
                 f'Latent Trajectory + Pattern Evolution',
                 fontsize=18, color='white', fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved dual visualization to {save_path}")


def create_dual_animation(model, vae, device, source_group, target_group,
                          save_path, n_frames=60, seed=42):
    """
    Create animated GIF showing latent trajectory AND pattern simultaneously.
    """
    torch.manual_seed(seed)
    
    model.eval()
    vae.eval()
    
    z_start = torch.randn(1, vae.config.latent_dim, device=device) * 0.8
    
    with torch.no_grad():
        trajectory = model(z_start, source_group, target_group, n_steps=n_frames)
    
    latent_points = []
    images = []
    
    with torch.no_grad():
        for i, z in enumerate(trajectory):
            latent_points.append(z[0].cpu().numpy())
            
            t = i / (n_frames - 1)
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            
            img_source = vae.decode(z, source_group)
            img_target = vae.decode(z, target_group)
            img = (1 - alpha) * img_source + alpha * img_target
            
            images.append(img[0, 0].cpu().numpy())
    
    latent_points = np.array(latent_points)
    
    source_color = LATTICE_INFO[source_group][1]
    target_color = LATTICE_INFO[target_group][1]
    
    # Create figure
    fig, (ax_latent, ax_pattern) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Setup latent space plot
    ax_latent.set_facecolor('#0a0a0a')
    ax_latent.set_xlim(latent_points[:, 0].min() - 0.3, latent_points[:, 0].max() + 0.3)
    ax_latent.set_ylim(latent_points[:, 1].min() - 0.3, latent_points[:, 1].max() + 0.3)
    ax_latent.set_xlabel('z₁', fontsize=12, color='white')
    ax_latent.set_ylabel('z₂', fontsize=12, color='white')
    ax_latent.set_title('Latent Space', fontsize=14, color='white', fontweight='bold')
    ax_latent.tick_params(colors='white')
    for spine in ax_latent.spines.values():
        spine.set_color('white')
    ax_latent.grid(True, alpha=0.2, color='white')
    ax_latent.set_aspect('equal')
    
    # Plot full trajectory as faint line
    ax_latent.plot(latent_points[:, 0], latent_points[:, 1], 
                  color='gray', alpha=0.3, linewidth=1)
    
    # Start/end markers
    ax_latent.scatter(latent_points[0, 0], latent_points[0, 1],
                     c=source_color, s=150, marker='o', edgecolors='white',
                     linewidths=2, zorder=10)
    ax_latent.scatter(latent_points[-1, 0], latent_points[-1, 1],
                     c=target_color, s=150, marker='*', edgecolors='white',
                     linewidths=2, zorder=10)
    
    # Current position marker
    current_point, = ax_latent.plot([], [], 'o', color='yellow', markersize=12,
                                    markeredgecolor='white', markeredgewidth=2)
    
    # Trail line
    trail_line, = ax_latent.plot([], [], color='#4ecdc4', linewidth=2, alpha=0.8)
    
    # Setup pattern plot
    ax_pattern.set_facecolor('#0a0a0a')
    ax_pattern.axis('off')
    im = ax_pattern.imshow(images[0], cmap='viridis', animated=True)
    pattern_title = ax_pattern.set_title(f'{source_group} → {target_group}: t=0.00',
                                         fontsize=14, color='white', fontweight='bold')
    
    def update(frame):
        t = frame / (n_frames - 1)
        
        # Update current point
        current_point.set_data([latent_points[frame, 0]], [latent_points[frame, 1]])
        
        # Update trail
        trail_line.set_data(latent_points[:frame+1, 0], latent_points[:frame+1, 1])
        
        # Update pattern
        im.set_array(images[frame])
        pattern_title.set_text(f'{source_group} → {target_group}: t={t:.2f}')
        
        return [current_point, trail_line, im, pattern_title]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    
    plt.tight_layout()
    
    writer = PillowWriter(fps=20)
    anim.save(save_path, writer=writer, facecolor='#0a0a0a')
    plt.close()
    
    print(f"Saved dual animation to {save_path}")


def create_transition_animation(model, vae, device, source_group, target_group,
                                 save_path, n_frames=60, seed=42):
    """Create an animated GIF of the phase transition."""
    
    torch.manual_seed(seed)
    
    model.eval()
    vae.eval()
    
    # Sample starting point
    z_start = torch.randn(1, vae.config.latent_dim, device=device) * 0.8
    
    # Get trajectory
    with torch.no_grad():
        trajectory = model(z_start, source_group, target_group, n_steps=n_frames)
    
    # Decode each point
    images = []
    
    with torch.no_grad():
        for i, z in enumerate(trajectory):
            t = i / (n_frames - 1)
            
            # Blend between source and target decoding
            img_source = vae.decode(z, source_group)
            img_target = vae.decode(z, target_group)
            
            # Smooth transition
            alpha = 0.5 * (1 - np.cos(np.pi * t))  # Smooth step
            img = (1 - alpha) * img_source + alpha * img_target
            
            images.append(img[0, 0].cpu().numpy())
    
    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.axis('off')
    
    im = ax.imshow(images[0], cmap='viridis', animated=True)
    title = ax.set_title(f'{source_group} → {target_group}: t=0.00',
                         fontsize=14, color='white', fontweight='bold')
    
    def update(frame):
        im.set_array(images[frame])
        t = frame / (n_frames - 1)
        title.set_text(f'{source_group} → {target_group}: t={t:.2f}')
        return [im, title]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(save_path, writer=writer, facecolor='#0a0a0a')
    plt.close()
    
    print(f"Saved animation to {save_path}")


def visualize_all_transitions_from_p1(model, vae, device, save_path, n_steps=10):
    """Visualize transitions from p1 (no symmetry) to all other groups."""
    
    model.eval()
    vae.eval()
    
    fig, axes = plt.subplots(17, n_steps, figsize=(n_steps * 1.5, 17 * 1.5))
    fig.patch.set_facecolor('#0a0a0a')
    
    z_start = torch.randn(1, vae.config.latent_dim, device=device) * 0.8
    
    with torch.no_grad():
        for group_idx, target_group in enumerate(ALL_17_GROUPS):
            # Get trajectory from p1 to target
            trajectory = model(z_start, 'p1', target_group, n_steps=n_steps)
            
            color = LATTICE_INFO[target_group][1]
            
            for step_idx in range(n_steps):
                ax = axes[group_idx, step_idx]
                
                z = trajectory[step_idx]
                t = step_idx / (n_steps - 1)
                
                # Decode with appropriate symmetry
                if t < 0.5:
                    img = vae.decode(z, 'p1')
                else:
                    img = vae.decode(z, target_group)
                
                ax.imshow(img[0, 0].cpu().numpy(), cmap='viridis')
                ax.axis('off')
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(1)
                
                # Labels
                if step_idx == 0:
                    ax.set_ylabel(target_group, fontsize=10, color=color,
                                 fontweight='bold', rotation=0, ha='right', va='center')
                if group_idx == 0:
                    ax.set_title(f't={t:.1f}', fontsize=9, color='white')
    
    fig.suptitle('Phase Transitions: p1 → All 17 Groups',
                 fontsize=18, color='white', fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved all transitions to {save_path}")


def load_vae(checkpoint_path: Path, device: torch.device):
    """Load pretrained VAE."""
    print(f"Loading VAE from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = SymmetryVAEConfig(
            resolution=128,
            latent_dim=2,
            hidden_dims=[64, 128, 256, 512]
        )
    
    vae = SymmetryInvariantVAE(config).to(device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    return vae, config


def find_latest_vae_checkpoint() -> Path:
    """Find most recent VAE checkpoint."""
    output_dir = Path("output")
    vae_dirs = sorted(output_dir.glob("symmetry_vae_17groups_*"), reverse=True)
    
    for vae_dir in vae_dirs:
        checkpoint = vae_dir / "best_model.pt"
        if checkpoint.exists():
            return checkpoint
    
    raise FileNotFoundError("No VAE checkpoint found")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Neural ODE for phase transitions')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                       help='Path to VAE checkpoint')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_pairs', type=int, default=5000,
                       help='Number of training pairs')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for ODE network')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/neural_ode_transitions_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load VAE
    if args.vae_checkpoint:
        vae_path = Path(args.vae_checkpoint)
    else:
        vae_path = find_latest_vae_checkpoint()
    
    vae, vae_config = load_vae(vae_path, device)
    print(f"VAE loaded (latent_dim={vae_config.latent_dim})")
    
    # Create Neural ODE model
    model = NeuralODETransition(
        latent_dim=vae_config.latent_dim,
        hidden_dim=args.hidden_dim,
        use_adjoint=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Neural ODE parameters: {n_params:,}")
    
    # Create dataset
    print("\n" + "="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    
    dataset = TransitionDataset(vae, device, num_pairs=args.num_pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Training
    print("\n" + "="*60)
    print("TRAINING NEURAL ODE")
    print("="*60)
    
    history = {'loss': []}
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        scheduler.step()
        
        history['loss'].append(loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss:.6f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, output_dir / 'best_model.pt')
    
    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Interesting transitions
    transitions = [
        ('p1', 'p4m'),   # No symmetry → full square symmetry
        ('p1', 'p6m'),   # No symmetry → full hexagonal symmetry
        ('p2', 'p4'),    # 2-fold → 4-fold rotation
        ('p3', 'p6'),    # 3-fold → 6-fold rotation
        ('pm', 'pmm'),   # Add perpendicular reflection
        ('p4', 'p4m'),   # Add reflections to square
        ('p3', 'p3m1'),  # Add reflections to hexagonal
        ('cmm', 'p4m'),  # Rectangular → Square
    ]
    
    for source, target in transitions:
        print(f"  Generating {source} → {target}...")
        
        # Static visualization
        visualize_transition(
            model, vae, device, source, target,
            output_dir / f'transition_{source}_to_{target}.png',
            n_frames=20
        )
        
        # DUAL visualization: latent space + patterns
        visualize_dual_transition(
            model, vae, device, source, target,
            output_dir / f'dual_{source}_to_{target}.png',
            n_frames=12
        )
        
        # DUAL animated GIF: latent trajectory + pattern
        create_dual_animation(
            model, vae, device, source, target,
            output_dir / f'dual_animation_{source}_to_{target}.gif',
            n_frames=50
        )
        
        # Simple animated GIF (pattern only)
        create_transition_animation(
            model, vae, device, source, target,
            output_dir / f'animation_{source}_to_{target}.gif',
            n_frames=40
        )
    
    # All transitions from p1
    visualize_all_transitions_from_p1(
        model, vae, device,
        output_dir / 'all_transitions_from_p1.png',
        n_steps=8
    )
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    ax.plot(history['loss'], color='#4ecdc4', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, color='white')
    ax.set_ylabel('Loss', fontsize=12, color='white')
    ax.set_title('Neural ODE Training Loss', fontsize=14, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=150,
                facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()


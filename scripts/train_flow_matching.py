#!/usr/bin/env python3
"""
Train Flow Matching for Crystallographic Phase Transitions.

State-of-the-art approach - faster and more stable than Neural ODE.

Usage:
    python scripts/train_flow_matching.py \
        --vae-checkpoint output/simple_vae_XXXX/best_model.pt \
        --epochs 100

Monitor training with separate dashboard:
    python scripts/dashboard_viewer.py --output-dir output/flow_matching_XXXX
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig
from src.models.flow_matching_transition import (
    FlowMatchingTransition,
    FlowMatchingConfig,
    FlowMatchingMetrics,
    ALL_17_GROUPS,
    GROUP_TO_IDX,
    IDX_TO_GROUP,
)
from src.dataset.transition_dataset import H5PatternDataset, TransitionDataset

# Colors for visualization
GROUP_COLORS = {
    'p1': '#FF6B6B', 'p2': '#4ECDC4', 'pm': '#45B7D1', 'pg': '#96CEB4',
    'cm': '#FFEAA7', 'pmm': '#DDA0DD', 'pmg': '#98D8C8', 'pgg': '#F7DC6F',
    'cmm': '#BB8FCE', 'p4': '#85C1E9', 'p4m': '#F8B500', 'p4g': '#00CED1',
    'p3': '#FF6347', 'p3m1': '#7B68EE', 'p31m': '#3CB371', 'p6': '#FF69B4',
    'p6m': '#00FA9A',
}


def load_vae(checkpoint_path: Path, device: torch.device):
    """Load pretrained VAE."""
    print(f"Loading VAE from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    
    if config is None:
        config = SimpleVAEConfig(latent_dim=64)
    elif not isinstance(config, SimpleVAEConfig):
        config = SimpleVAEConfig(latent_dim=getattr(config, 'latent_dim', 64))
    
    vae = SimpleVAE(config).to(device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    for param in vae.parameters():
        param.requires_grad = False
    
    print(f"VAE loaded: latent_dim={config.latent_dim}")
    return vae, config


def train_epoch(model, dataloader, optimizer, device, scaler, use_amp=True):
    """Train for one epoch."""
    model.train()
    
    total_losses = {'loss': 0.0, 'flow_loss': 0.0, 'velocity_reg': 0.0}
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (z_source, z_target, source_idx, target_idx) in enumerate(pbar):
        z_source = z_source.to(device, non_blocking=True)
        z_target = z_target.to(device, non_blocking=True)
        source_idx = source_idx.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=use_amp):
            losses = model.compute_loss(z_source, z_target, source_idx, target_idx)
            loss = losses['loss']
        
        # Skip NaN
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate
        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n_batches += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    
    # Average
    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    
    return total_losses


# Global cache for visualization
_viz_cache = {'pca': None, 'latents': None}

@torch.no_grad()
def visualize_trajectories(model, vae, h5_dataset, device, save_path, epoch, use_amp=True, 
                           transition_dataset=None):
    """Visualize latent space with sample trajectories (optimized)."""
    model.eval()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Use precomputed latents from transition dataset if available
    if transition_dataset is not None and hasattr(transition_dataset, 'latents'):
        all_latents = {}
        for group_idx in range(17):
            latents_list = transition_dataset.latents.get(group_idx, [])
            if latents_list:
                # Take first 15 latents max
                zs = [item[1] if isinstance(item, tuple) else item for item in latents_list[:15]]
                all_latents[group_idx] = torch.stack(zs).numpy()
    else:
        # Fallback: encode a few patterns per group
        vae.eval()
        all_latents = {}
        for group_idx, group_name in enumerate(ALL_17_GROUPS):
            indices = h5_dataset.indices_by_group.get(group_idx, [])
            if len(indices) < 3:
                continue
            
            patterns = torch.stack([
                h5_dataset.get_pattern_by_group(group_idx, i) 
                for i in range(min(10, len(indices)))
            ]).to(device)
            
            with autocast('cuda', enabled=use_amp):
                z = vae.encode(patterns, group_name)
            all_latents[group_idx] = z.float().cpu().numpy()
    
    if not all_latents:
        plt.close()
        return
    
    # PCA to 2D (cache it)
    latent_dim = list(all_latents.values())[0].shape[1]
    
    if latent_dim > 2:
        all_z = np.concatenate(list(all_latents.values()), axis=0)
        
        if _viz_cache['pca'] is None:
            from sklearn.decomposition import PCA
            _viz_cache['pca'] = PCA(n_components=2)
            _viz_cache['pca'].fit(all_z)
        
        pca = _viz_cache['pca']
        all_latents = {k: pca.transform(v) for k, v in all_latents.items()}
    
    # Plot each group
    for group_idx, z_2d in all_latents.items():
        group_name = IDX_TO_GROUP[group_idx]
        color = GROUP_COLORS.get(group_name, '#888888')
        ax.scatter(z_2d[:, 0], z_2d[:, 1], c=color, label=group_name, alpha=0.7, s=25)
    
    # Draw arrows for transitions (simple lines, no model calls)
    transitions = [('p1', 'p6m'), ('p2', 'p4'), ('pm', 'pmm'), ('p3', 'p6')]
    for source, target in transitions:
        src_idx, tgt_idx = GROUP_TO_IDX[source], GROUP_TO_IDX[target]
        if src_idx in all_latents and tgt_idx in all_latents:
            start = all_latents[src_idx][0]
            end = all_latents[tgt_idx][0]
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color='white', alpha=0.4, lw=1.5))
    
    ax.set_xlabel('PCA 1', color='white', fontsize=11)
    ax.set_ylabel('PCA 2', color='white', fontsize=11)
    ax.set_title(f'Latent Space - Epoch {epoch}', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white',
              labelcolor='white', fontsize=7, ncol=3)
    ax.grid(True, alpha=0.15, color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


@torch.no_grad()
def generate_transition_gif(model, vae, h5_dataset, device, source, target, save_dir, use_amp=True):
    """Generate transition visualization (optimized - static strip only, no GIF)."""
    model.eval()
    vae.eval()
    
    source_idx = GROUP_TO_IDX[source]
    target_idx = GROUP_TO_IDX[target]
    
    # Get a sample pattern
    indices = h5_dataset.indices_by_group.get(source_idx, [])
    if len(indices) == 0:
        return
    
    pattern = h5_dataset.get_pattern_by_group(source_idx, 0).unsqueeze(0).to(device)
    
    with autocast('cuda', enabled=use_amp):
        z_start = vae.encode(pattern, source)
    
    src_t = torch.tensor([source_idx], device=device)
    tgt_t = torch.tensor([target_idx], device=device)
    
    # Get trajectory (fewer frames for speed)
    n_frames = 16
    trajectory = model.sample_trajectory(z_start, src_t, tgt_t, n_steps=n_frames)
    
    # Decode only key frames
    n_show = 6
    key_indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
    
    decoded = []
    for idx in key_indices:
        with autocast('cuda', enabled=use_amp):
            img = vae.decode(trajectory[idx].float(), target)
        decoded.append(img[0].float().cpu().clamp(0, 1))
    
    # Save static strip only (skip slow GIF)
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2.5))
    fig.patch.set_facecolor('#0a0a0a')
    
    for i, img in enumerate(decoded):
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].axis('off')
        t_val = key_indices[i] / (n_frames - 1)
        axes[i].set_title(f't={t_val:.1f}', color='white', fontsize=10)
    
    fig.suptitle(f'{source} → {target}', fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'transition_{source}_{target}.png', dpi=100, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


@torch.no_grad()
def generate_full_visualizations(model, vae, h5_dataset, device, output_dir, epoch, use_amp=True, 
                                  n_samples=2, all_pairs=False, n_frames=60):
    """Generate complete visualizations: transitions, latent space, GIFs.
    
    Args:
        all_pairs: If True, generate all 17x17 transitions (for final epoch)
        n_frames: Number of frames in animations (default 60 for smooth transitions)
    """
    from sklearn.decomposition import PCA
    
    viz_dir = output_dir / 'full_visualizations' / f'epoch_{epoch:03d}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    vae.eval()
    
    # Define transitions to visualize
    if all_pairs:
        # All 17x17 = 289 pairs at the end
        transitions = [(src, tgt) for src in ALL_17_GROUPS for tgt in ALL_17_GROUPS if src != tgt]
        print(f"  Generating {len(transitions)} transition pairs...")
    else:
        # Representative transitions during training
        transitions = [
            # Minimal → Maximal symmetry
            ('p1', 'p6m'), ('p1', 'p4m'), ('p1', 'pmm'),
            # Rotation increases
            ('p1', 'p2'), ('p2', 'p4'), ('p3', 'p6'),
            # Adding reflections
            ('p2', 'pmm'), ('p4', 'p4m'), ('p6', 'p6m'), ('p3', 'p3m1'),
            # Reverse transitions
            ('p6m', 'p1'), ('p4m', 'p2'),
        ]
    
    # 1. Generate transition GIFs and strips
    trans_dir = viz_dir / 'transitions'
    trans_dir.mkdir(exist_ok=True)
    
    n_show = 12  # More frames in strip for detailed view
    fps = 5  # Slower animation (5 fps = smooth transitions)
    
    for source, target in tqdm(transitions, desc="Transitions", leave=False):
        try:
            source_idx = GROUP_TO_IDX[source]
            target_idx = GROUP_TO_IDX[target]
            
            indices = h5_dataset.indices_by_group.get(source_idx, [])
            if len(indices) == 0:
                continue
            
            # For all pairs, only 1 sample; otherwise n_samples
            samples_to_gen = 1 if all_pairs else min(n_samples, len(indices))
            
            for sample_i in range(samples_to_gen):
                pattern = h5_dataset.get_pattern_by_group(source_idx, sample_i).unsqueeze(0).to(device)
                
                with autocast('cuda', enabled=use_amp):
                    z_start = vae.encode(pattern, source)
                
                src_t = torch.tensor([source_idx], device=device)
                tgt_t = torch.tensor([target_idx], device=device)
                
                trajectory = model.sample_trajectory(z_start, src_t, tgt_t, n_steps=n_frames)
                
                # Decode frames
                decoded = []
                for i in range(n_frames):
                    with autocast('cuda', enabled=use_amp):
                        img = vae.decode(trajectory[i].float(), target)
                    decoded.append(img[0].float().cpu().clamp(0, 1))
                
                # Save strip with more frames and better styling
                key_indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
                
                fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2.2, 3))
                fig.patch.set_facecolor('#0a0a0a')
                
                for i, idx in enumerate(key_indices):
                    axes[i].imshow(decoded[idx].permute(1, 2, 0).numpy())
                    axes[i].axis('off')
                    t = idx / (n_frames - 1)
                    axes[i].set_title(f't={t:.2f}', color='#888', fontsize=10, pad=4)
                    # Add border to first and last frames
                    if i == 0:
                        for spine in axes[i].spines.values():
                            spine.set_edgecolor('#FF6B6B')
                            spine.set_linewidth(3)
                            spine.set_visible(True)
                    elif i == n_show - 1:
                        for spine in axes[i].spines.values():
                            spine.set_edgecolor('#4ECDC4')
                            spine.set_linewidth(3)
                            spine.set_visible(True)
                
                fig.suptitle(f'{source} → {target}', fontsize=15, color='white', fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.92])
                
                suffix = '' if all_pairs else f'_s{sample_i+1}'
                plt.savefig(trans_dir / f'{source}_to_{target}{suffix}.png', 
                           dpi=120, facecolor='#0a0a0a', bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
                # Save GIF (slower, smoother)
                if sample_i == 0:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.axis('off')
                    
                    im = ax.imshow(decoded[0].permute(1, 2, 0).numpy())
                    title = ax.set_title(f'{source}→{target}: t=0.00', color='white', fontsize=12)
                    
                    def make_update(decoded_frames, src, tgt, n_f):
                        def update(frame):
                            im.set_data(decoded_frames[frame].permute(1, 2, 0).numpy())
                            t = frame / (n_f - 1)
                            title.set_text(f'{src}→{tgt}: t={t:.2f}')
                            return [im, title]
                        return update
                    
                    anim = FuncAnimation(fig, make_update(decoded, source, target, n_frames), 
                                        frames=n_frames, interval=200, blit=True)
                    anim.save(trans_dir / f'{source}_to_{target}.gif', 
                             writer=PillowWriter(fps=fps), savefig_kwargs={'facecolor': '#0a0a0a'})
                    plt.close()
                    
                    # Save latent trajectory visualization
                    traj_np = trajectory[:, 0].cpu().numpy()
                    
                    fig, ax = plt.subplots(figsize=(8, 7))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.set_facecolor('#0f0f1a')
                    
                    # Use first 2 dims or PCA projection
                    if traj_np.shape[1] <= 2:
                        traj_2d = traj_np[:, :2]
                    else:
                        # Use PCA for consistent projection of trajectory
                        from sklearn.decomposition import PCA
                        pca_traj = PCA(n_components=2)
                        traj_2d = pca_traj.fit_transform(traj_np)
                    
                    # Add margin to axes
                    margin = 0.15
                    x_range = traj_2d[:, 0].max() - traj_2d[:, 0].min()
                    y_range = traj_2d[:, 1].max() - traj_2d[:, 1].min()
                    ax.set_xlim(traj_2d[:, 0].min() - margin * x_range, traj_2d[:, 0].max() + margin * x_range)
                    ax.set_ylim(traj_2d[:, 1].min() - margin * y_range, traj_2d[:, 1].max() + margin * y_range)
                    
                    # Plot trajectory with color gradient (thicker line)
                    colors = plt.cm.plasma(np.linspace(0, 1, len(traj_2d)))
                    for i in range(len(traj_2d) - 1):
                        ax.plot(traj_2d[i:i+2, 0], traj_2d[i:i+2, 1], 
                               color=colors[i], linewidth=4, alpha=0.9, solid_capstyle='round')
                    
                    # Mark fewer points to avoid clutter
                    n_markers = min(8, len(traj_2d))
                    scatter_indices = np.linspace(0, len(traj_2d)-1, n_markers, dtype=int)
                    for idx in scatter_indices:
                        t_val = idx / (len(traj_2d) - 1)
                        ax.scatter(traj_2d[idx, 0], traj_2d[idx, 1], 
                                  c=[colors[idx]], s=120, edgecolors='white', linewidths=2, zorder=10)
                        # Offset labels to avoid overlap
                        offset_x = 8 if idx < len(traj_2d) // 2 else -35
                        ax.annotate(f't={t_val:.1f}', xy=(traj_2d[idx, 0], traj_2d[idx, 1]),
                                   xytext=(offset_x, 8), textcoords='offset points',
                                   color='white', fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0a0a', alpha=0.8))
                    
                    # Start and end markers (larger, more prominent)
                    ax.scatter(traj_2d[0, 0], traj_2d[0, 1], c='#FF6B6B', s=300, 
                              marker='o', edgecolors='white', linewidths=3, zorder=25, label=f'Start: {source}')
                    ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='#4ECDC4', s=350, 
                              marker='*', edgecolors='white', linewidths=3, zorder=25, label=f'End: {target}')
                    
                    ax.set_xlabel('Latent Dimension 1', color='white', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Latent Dimension 2', color='white', fontsize=12, fontweight='bold')
                    ax.set_title(f'Latent Space Trajectory: {source} → {target}', 
                                color='white', fontsize=16, fontweight='bold', pad=15)
                    ax.tick_params(colors='white', labelsize=10)
                    for spine in ax.spines.values():
                        spine.set_color('#333')
                    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444', 
                             labelcolor='white', fontsize=10, framealpha=0.95)
                    ax.grid(True, alpha=0.15, color='white', linestyle='--')
                    
                    plt.tight_layout(pad=1.5)
                    plt.savefig(trans_dir / f'{source}_to_{target}_latent.png', 
                               dpi=120, facecolor='#0a0a0a', bbox_inches='tight', pad_inches=0.3)
                    plt.close()
                    
        except Exception as e:
            continue
    
    # 2. Latent space with trajectories
    try:
        all_latents = {}
        for group_idx in range(17):
            indices = h5_dataset.indices_by_group.get(group_idx, [])
            if len(indices) < 3:
                continue
            
            patterns = []
            for i in range(min(20, len(indices))):
                patterns.append(h5_dataset.get_pattern_by_group(group_idx, i))
            
            patterns = torch.stack(patterns).to(device)
            group_name = IDX_TO_GROUP[group_idx]
            with autocast('cuda', enabled=use_amp):
                z = vae.encode(patterns, group_name)
            all_latents[group_idx] = z.float().cpu().numpy()
        
        if all_latents:
            all_z = np.concatenate(list(all_latents.values()), axis=0)
            
            if all_z.shape[1] <= 2:
                # Direct 2D latent space - no projection needed
                latents_2d = {k: v[:, :2] for k, v in all_latents.items()}
                proj_func = lambda x: x[:, :2]
            else:
                # Use UMAP for better cluster visualization
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                                       metric='euclidean', random_state=42)
                    all_z_2d = reducer.fit_transform(all_z)
                    
                    # Split back into groups
                    idx = 0
                    latents_2d = {}
                    for k, v in all_latents.items():
                        latents_2d[k] = all_z_2d[idx:idx+len(v)]
                        idx += len(v)
                    
                    def proj_func(x):
                        return reducer.transform(x)
                        
                except ImportError:
                    # Fallback to PCA if UMAP not available
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    all_z_scaled = scaler.fit_transform(all_z)
                    
                    pca = PCA(n_components=2)
                    pca.fit(all_z_scaled)
                    
                    latents_2d = {}
                    for k, v in all_latents.items():
                        v_scaled = scaler.transform(v)
                        latents_2d[k] = pca.transform(v_scaled)
                    
                    def proj_func(x):
                        return pca.transform(scaler.transform(x))
            
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.patch.set_facecolor('#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            
            for group_idx, z_2d in latents_2d.items():
                group_name = IDX_TO_GROUP[group_idx]
                color = GROUP_COLORS.get(group_name, '#888888')
                ax.scatter(z_2d[:, 0], z_2d[:, 1], c=color, label=group_name, 
                          alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
            
            # Draw trajectories
            for source, target, traj_color in [('p1', 'p6m', '#FF6B6B'), ('p2', 'p4', '#4ECDC4'), ('p3', 'p6', '#FFD93D')]:
                source_idx = GROUP_TO_IDX[source]
                if source_idx not in all_latents:
                    continue
                
                z_start = torch.tensor(all_latents[source_idx][:1], device=device, dtype=torch.float32)
                src_t = torch.tensor([source_idx], device=device)
                tgt_t = torch.tensor([GROUP_TO_IDX[target]], device=device)
                
                traj = model.sample_trajectory(z_start, src_t, tgt_t, n_steps=20)
                traj_2d = proj_func(traj[:, 0].cpu().numpy())
                
                ax.plot(traj_2d[:, 0], traj_2d[:, 1], color=traj_color, linewidth=3, alpha=0.9)
                ax.scatter(traj_2d[0, 0], traj_2d[0, 1], c=traj_color, s=150, marker='o', 
                          edgecolors='white', linewidths=2, zorder=10)
                ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c=traj_color, s=200, marker='*', 
                          edgecolors='white', linewidths=2, zorder=10)
                # Add label
                mid = len(traj_2d) // 2
                ax.annotate(f'{source}→{target}', xy=traj_2d[mid], color=traj_color, 
                           fontsize=10, fontweight='bold', 
                           bbox=dict(boxstyle='round', facecolor='#0a0a0a', alpha=0.7))
            
            ax.set_xlabel('UMAP Dimension 1', color='white', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', color='white', fontsize=12)
            ax.set_title(f'Latent Space (UMAP) - Epoch {epoch}', color='white', fontsize=16, fontweight='bold')
            ax.tick_params(colors='white', labelsize=10)
            ax.set_aspect('equal', adjustable='datalim')  # Equal aspect ratio
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white', 
                     labelcolor='white', fontsize=8, ncol=3)
            ax.grid(True, alpha=0.3, color='white', linestyle='--')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'latent_space.png', dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"  Latent space viz failed: {e}")
    
    # Update live state for dashboard
    state_file = output_dir / 'live_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            state['latest_viz_epoch'] = epoch
            state['viz_dir'] = str(viz_dir)
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass
    
    print(f"  ✅ Saved to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Flow Matching for Crystallographic Phase Transitions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to trained VAE checkpoint')
    
    # Data
    parser.add_argument('--data-path', type=str, 
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for checkpoints and visualizations')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--pairs-per-epoch', type=int, default=20000,
                       help='Number of transition pairs per epoch')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension of flow network')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of layers in flow network')
    
    # Visualization
    parser.add_argument('--viz-interval', type=int, default=10,
                       help='Epochs between basic visualizations')
    parser.add_argument('--full-viz-interval', type=int, default=50,
                       help='Epochs between full visualizations (GIFs, strips)')
    parser.add_argument('--n-frames', type=int, default=60,
                       help='Number of frames in transition animations')
    parser.add_argument('--viz-samples', type=int, default=2,
                       help='Number of samples per transition during training')
    parser.add_argument('--final-all-pairs', action='store_true', default=True,
                       help='Generate all 17x17 transitions at end')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and torch.cuda.is_available()
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    print(f"Using device: {device}")
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'flow_matching_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Live state file for external dashboard
    live_state_path = output_dir / 'live_state.json'
    
    # Load VAE
    vae, vae_config = load_vae(Path(args.vae_checkpoint), device)
    
    # Load dataset
    print("\nLoading dataset...")
    h5_dataset = H5PatternDataset(args.data_path, split='train')
    print(f"Dataset: {len(h5_dataset)} patterns")
    
    # Create transition dataset
    print("\nCreating transition pairs...")
    transition_dataset = TransitionDataset(
        vae=vae,
        h5_dataset=h5_dataset,
        device=device,
        pairs_per_epoch=args.pairs_per_epoch,
        seed=42,
    )
    
    dataloader = DataLoader(
        transition_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create Flow Matching model
    config = FlowMatchingConfig(
        latent_dim=vae_config.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    
    model = FlowMatchingTransition(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flow Matching parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'model': 'FlowMatching',
            'vae_checkpoint': str(args.vae_checkpoint),
            'latent_dim': config.latent_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        }, f, indent=2)
    
    # Training
    history = {'loss': [], 'flow_loss': [], 'velocity_reg': []}
    best_loss = float('inf')
    
    print("\n" + "="*60)
    print("TRAINING FLOW MATCHING (State-of-the-Art)")
    print("="*60)
    
    
    train_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Regenerate pairs for variety
        if epoch > 1:
            transition_dataset.regenerate_pairs()
        
        # Train
        losses = train_epoch(model, dataloader, optimizer, device, scaler, use_amp)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        for k, v in losses.items():
            history[k].append(v)
        
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {losses['loss']:.6f} | "
              f"Flow: {losses['flow_loss']:.6f} | Time: {epoch_time:.1f}s")
        
        # Save live state for external dashboard
        live_state = {
            'epoch': epoch,
            'total_epochs': args.epochs,
            'loss': losses['loss'],
            'flow_loss': losses['flow_loss'],
            'best_loss': best_loss,
            'history': history,
            'is_training': True,
            'output_dir': str(output_dir),
        }
        with open(live_state_path, 'w') as f:
            json.dump(live_state, f)
        
        # Save best
        if losses['loss'] < best_loss:
            best_loss = losses['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': best_loss,
            }, output_dir / 'best_model.pt')
        
        # Visualizations
        if epoch % args.viz_interval == 0:
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            try:
                visualize_trajectories(
                    model, vae, h5_dataset, device,
                    viz_dir / f'latent_epoch_{epoch:03d}.png',
                    epoch, use_amp, transition_dataset
                )
            except Exception as e:
                print(f"  Visualization failed: {e}")
        
        # Full visualizations periodically
        if epoch % args.full_viz_interval == 0 and epoch > 0:
            print(f"\n📊 Generating full visualizations at epoch {epoch}...")
            generate_full_visualizations(
                model, vae, h5_dataset, device, output_dir, 
                epoch, use_amp, n_samples=args.viz_samples, 
                all_pairs=False, n_frames=args.n_frames
            )
    
    # Final save
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate complete visualizations at the end (all 17x17 pairs if enabled)
    print("\n📊 Generating final visualizations...")
    generate_full_visualizations(
        model, vae, h5_dataset, device, output_dir,
        epoch=args.epochs, use_amp=use_amp, 
        n_samples=3, all_pairs=args.final_all_pairs, n_frames=args.n_frames
    )
    
    # Training curves
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    ax.plot(history['loss'], 'c-', label='Total Loss', linewidth=2)
    ax.plot(history['flow_loss'], 'm-', label='Flow Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Loss', color='white')
    ax.set_title('Flow Matching Training', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, facecolor='#0a0a0a')
    plt.close()
    
    total_time = time.time() - train_start
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()


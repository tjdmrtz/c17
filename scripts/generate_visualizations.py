#!/usr/bin/env python3
"""
Generate Complete Visualizations for Trained Flow Matching Model.

Creates:
- Transition strips for all group pairs
- Animated GIFs
- Latent space with trajectories
- Reconstruction comparisons
- Group-by-group galleries

Usage:
    python scripts/generate_visualizations.py --model-dir output/flow_matching_XXXX
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from torch.amp import autocast
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig
from src.models.flow_matching_transition import (
    FlowMatchingTransition,
    FlowMatchingConfig,
    ALL_17_GROUPS,
    GROUP_TO_IDX,
    IDX_TO_GROUP,
)
from src.dataset.transition_dataset import H5PatternDataset

# Colors
GROUP_COLORS = {
    'p1': '#FF6B6B', 'p2': '#4ECDC4', 'pm': '#45B7D1', 'pg': '#96CEB4',
    'cm': '#FFEAA7', 'pmm': '#DDA0DD', 'pmg': '#98D8C8', 'pgg': '#F7DC6F',
    'cmm': '#BB8FCE', 'p4': '#85C1E9', 'p4m': '#F8B500', 'p4g': '#00CED1',
    'p3': '#FF6347', 'p3m1': '#7B68EE', 'p31m': '#3CB371', 'p6': '#FF69B4',
    'p6m': '#00FA9A',
}


def load_models(model_dir: Path, device: torch.device):
    """Load VAE and Flow Matching models."""
    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    vae_path = Path(config['vae_checkpoint'])
    
    # Load VAE
    print(f"Loading VAE from {vae_path}")
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae_config = vae_ckpt.get('config', SimpleVAEConfig(latent_dim=64))
    if not isinstance(vae_config, SimpleVAEConfig):
        vae_config = SimpleVAEConfig(latent_dim=getattr(vae_config, 'latent_dim', 64))
    
    vae = SimpleVAE(vae_config).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    
    # Load Flow Matching
    print(f"Loading Flow Matching model")
    fm_ckpt = torch.load(model_dir / 'best_model.pt', map_location=device, weights_only=False)
    fm_config = fm_ckpt.get('config', FlowMatchingConfig(latent_dim=vae_config.latent_dim))
    
    model = FlowMatchingTransition(fm_config).to(device)
    model.load_state_dict(fm_ckpt['model_state_dict'])
    model.eval()
    
    return vae, model, vae_config


@torch.no_grad()
def generate_transition_strip(model, vae, z_start, source_idx, target_idx, 
                               source_name, target_name, n_frames=20, device='cuda'):
    """Generate a strip of transition frames."""
    src_t = torch.tensor([source_idx], device=device)
    tgt_t = torch.tensor([target_idx], device=device)
    
    # Get trajectory
    trajectory = model.sample_trajectory(z_start, src_t, tgt_t, n_steps=n_frames)
    
    # Decode all frames
    decoded = []
    for i in range(n_frames):
        with autocast('cuda'):
            img = vae.decode(trajectory[i].float(), target_name)
        decoded.append(img[0].float().cpu().clamp(0, 1))
    
    return decoded, trajectory


@torch.no_grad()
def generate_all_transitions(model, vae, h5_dataset, device, output_dir, n_samples=3):
    """Generate transition visualizations for interesting group pairs."""
    print("\n📊 Generating transition visualizations...")
    
    transitions_dir = output_dir / 'transitions'
    transitions_dir.mkdir(exist_ok=True)
    
    # Interesting transitions based on symmetry relationships
    key_transitions = [
        # Minimal to maximal symmetry
        ('p1', 'p6m'), ('p1', 'p4m'), ('p1', 'pmm'),
        # Rotation increases
        ('p1', 'p2'), ('p2', 'p4'), ('p3', 'p6'),
        # Adding reflections
        ('p2', 'pmm'), ('p4', 'p4m'), ('p6', 'p6m'), ('p3', 'p3m1'),
        # Glide reflections
        ('pm', 'pg'), ('pmm', 'pmg'), ('p4m', 'p4g'),
        # Centered lattices
        ('pm', 'cm'), ('pmm', 'cmm'),
    ]
    
    for source, target in tqdm(key_transitions, desc="Generating transitions"):
        source_idx = GROUP_TO_IDX[source]
        target_idx = GROUP_TO_IDX[target]
        
        indices = h5_dataset.indices_by_group.get(source_idx, [])
        if len(indices) == 0:
            continue
        
        # Get multiple samples
        for sample_i in range(min(n_samples, len(indices))):
            try:
                pattern = h5_dataset.get_pattern_by_group(source_idx, sample_i).unsqueeze(0).to(device)
                
                with autocast('cuda'):
                    z_start = vae.encode(pattern, source)
                
                # Generate transition
                n_frames = 24
                decoded, trajectory = generate_transition_strip(
                    model, vae, z_start, source_idx, target_idx, source, target, n_frames, device
                )
                
                # Create strip image
                n_show = 8
                key_indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
                
                fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2.5, 3))
                fig.patch.set_facecolor('#0a0a0a')
                
                for i, idx in enumerate(key_indices):
                    axes[i].imshow(decoded[idx].permute(1, 2, 0).numpy())
                    axes[i].axis('off')
                    t = idx / (n_frames - 1)
                    axes[i].set_title(f't={t:.2f}', color='white', fontsize=11)
                
                fig.suptitle(f'{source} → {target} (sample {sample_i+1})', 
                            fontsize=16, color='white', fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(transitions_dir / f'{source}_to_{target}_sample{sample_i+1}.png', 
                           dpi=120, facecolor='#0a0a0a', bbox_inches='tight')
                plt.close()
                
                # Create animated GIF (only for first sample)
                if sample_i == 0:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.axis('off')
                    
                    im = ax.imshow(decoded[0].permute(1, 2, 0).numpy())
                    title = ax.set_title(f'{source} → {target}: t=0.00', color='white', fontsize=14, pad=10)
                    
                    def update(frame):
                        im.set_data(decoded[frame].permute(1, 2, 0).numpy())
                        t = frame / (n_frames - 1)
                        title.set_text(f'{source} → {target}: t={t:.2f}')
                        return [im, title]
                    
                    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=True)
                    writer = PillowWriter(fps=12)
                    anim.save(transitions_dir / f'{source}_to_{target}.gif', writer=writer,
                             savefig_kwargs={'facecolor': '#0a0a0a'})
                    plt.close()
                    
            except Exception as e:
                print(f"  Warning: {source} → {target} sample {sample_i+1} failed: {e}")
                continue
    
    print(f"  Saved to {transitions_dir}")


@torch.no_grad()
def generate_latent_space_visualization(model, vae, h5_dataset, device, output_dir):
    """Generate detailed latent space visualization with trajectories."""
    print("\n🎯 Generating latent space visualization...")
    
    viz_dir = output_dir / 'latent_space'
    viz_dir.mkdir(exist_ok=True)
    
    # Collect latents from each group
    all_latents = {}
    all_patterns = {}
    
    for group_idx, group_name in enumerate(ALL_17_GROUPS):
        indices = h5_dataset.indices_by_group.get(group_idx, [])
        if len(indices) < 3:
            continue
        
        patterns = []
        for i in range(min(30, len(indices))):
            pattern = h5_dataset.get_pattern_by_group(group_idx, i)
            patterns.append(pattern)
        
        patterns = torch.stack(patterns).to(device)
        with autocast('cuda'):
            z = vae.encode(patterns, group_name)
        
        all_latents[group_idx] = z.float().cpu().numpy()
        all_patterns[group_idx] = patterns.cpu()
    
    # Project to 2D
    all_z = np.concatenate(list(all_latents.values()), axis=0)
    
    if all_z.shape[1] <= 2:
        # Direct 2D latent space - no projection needed
        latents_2d = {k: v[:, :2] for k, v in all_latents.items()}
        proj_func = lambda x: x[:, :2]
        print("  Using direct 2D latent space")
    else:
        # Use UMAP for better cluster visualization
        try:
            import umap
            print("  Using UMAP projection (preserves cluster structure)...")
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
            # Fallback to PCA
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            print("  UMAP not available, using PCA...")
            
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
    
    # Large visualization
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    
    # Main latent space plot
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#0a0a0a')
    
    for group_idx, z_2d in latents_2d.items():
        group_name = IDX_TO_GROUP[group_idx]
        color = GROUP_COLORS.get(group_name, '#888888')
        ax_main.scatter(z_2d[:, 0], z_2d[:, 1], c=color, label=group_name, 
                       alpha=0.7, s=50, edgecolors='white', linewidths=0.3)
    
    # Draw trajectories
    trajectories_to_draw = [
        ('p1', 'p6m', 'white'),
        ('p2', 'p4', 'cyan'),
        ('p3', 'p6', 'magenta'),
        ('pm', 'pmm', 'yellow'),
    ]
    
    for source, target, color in trajectories_to_draw:
        source_idx = GROUP_TO_IDX[source]
        target_idx = GROUP_TO_IDX[target]
        
        if source_idx not in latents_2d or target_idx not in latents_2d:
            continue
        
        # Get trajectory in original space and project
        z_start_orig = torch.tensor(all_latents[source_idx][:1], device=device, dtype=torch.float32)
        
        # Need to get trajectory in original 64D space then project
        src_t = torch.tensor([source_idx], device=device)
        tgt_t = torch.tensor([target_idx], device=device)
        trajectory = model.sample_trajectory(z_start_orig, src_t, tgt_t, n_steps=30)
        traj_np = trajectory[:, 0].cpu().numpy()
        traj_2d = proj_func(traj_np)
        
        ax_main.plot(traj_2d[:, 0], traj_2d[:, 1], color=color, linewidth=2.5, alpha=0.8)
        ax_main.scatter(traj_2d[0, 0], traj_2d[0, 1], c=color, s=150, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
        ax_main.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c=color, s=150, marker='*', 
                       edgecolors='white', linewidths=2, zorder=10)
        ax_main.annotate(f'{source}→{target}', xy=traj_2d[len(traj_2d)//2], 
                        color=color, fontsize=10, fontweight='bold')
    
    ax_main.set_xlabel('PCA 1', color='white', fontsize=14)
    ax_main.set_ylabel('PCA 2', color='white', fontsize=14)
    ax_main.set_title('Latent Space with Flow Matching Trajectories', 
                      color='white', fontsize=18, fontweight='bold', pad=20)
    ax_main.tick_params(colors='white')
    for spine in ax_main.spines.values():
        spine.set_color('white')
    ax_main.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white',
                  labelcolor='white', fontsize=9, ncol=3)
    ax_main.grid(True, alpha=0.2, color='white')
    
    # Group centroids subplot
    ax_centroids = fig.add_subplot(gs[1, 0])
    ax_centroids.set_facecolor('#0a0a0a')
    
    centroids = []
    centroid_names = []
    for group_idx in range(17):
        if group_idx in latents_2d:
            centroid = latents_2d[group_idx].mean(axis=0)
            centroids.append(centroid)
            centroid_names.append(IDX_TO_GROUP[group_idx])
    
    centroids = np.array(centroids)
    for i, name in enumerate(centroid_names):
        color = GROUP_COLORS.get(name, '#888888')
        ax_centroids.scatter(centroids[i, 0], centroids[i, 1], c=color, s=200, 
                            edgecolors='white', linewidths=2)
        ax_centroids.annotate(name, xy=centroids[i], xytext=(5, 5), 
                             textcoords='offset points', color='white', fontsize=9)
    
    ax_centroids.set_title('Group Centroids', color='white', fontsize=14)
    ax_centroids.tick_params(colors='white')
    for spine in ax_centroids.spines.values():
        spine.set_color('white')
    ax_centroids.grid(True, alpha=0.2)
    
    # Variance by group
    ax_var = fig.add_subplot(gs[1, 1])
    ax_var.set_facecolor('#0a0a0a')
    
    variances = []
    var_names = []
    var_colors = []
    for group_idx in range(17):
        if group_idx in latents_2d:
            var = latents_2d[group_idx].var()
            variances.append(var)
            name = IDX_TO_GROUP[group_idx]
            var_names.append(name)
            var_colors.append(GROUP_COLORS.get(name, '#888888'))
    
    ax_var.bar(range(len(variances)), variances, color=var_colors)
    ax_var.set_xticks(range(len(var_names)))
    ax_var.set_xticklabels(var_names, rotation=45, ha='right', color='white', fontsize=9)
    ax_var.set_title('Latent Variance by Group', color='white', fontsize=14)
    ax_var.set_ylabel('Variance', color='white')
    ax_var.tick_params(colors='white')
    for spine in ax_var.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'latent_space_complete.png', dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {viz_dir}")


@torch.no_grad()
def generate_reconstruction_gallery(vae, h5_dataset, device, output_dir):
    """Generate gallery showing VAE reconstructions for each group."""
    print("\n🖼️ Generating reconstruction gallery...")
    
    gallery_dir = output_dir / 'reconstructions'
    gallery_dir.mkdir(exist_ok=True)
    
    n_samples = 5
    
    fig, axes = plt.subplots(17, n_samples * 2, figsize=(n_samples * 4, 17 * 2))
    fig.patch.set_facecolor('#0a0a0a')
    
    for group_idx, group_name in enumerate(ALL_17_GROUPS):
        indices = h5_dataset.indices_by_group.get(group_idx, [])
        if len(indices) == 0:
            continue
        
        for i in range(min(n_samples, len(indices))):
            pattern = h5_dataset.get_pattern_by_group(group_idx, i).unsqueeze(0).to(device)
            
            with autocast('cuda'):
                output = vae(pattern, group_name)
                recon = output['recon']
            
            orig = pattern[0].cpu().float().permute(1, 2, 0).numpy()
            rec = recon[0].cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
            
            # Original
            axes[group_idx, i * 2].imshow(orig)
            axes[group_idx, i * 2].axis('off')
            if i == 0:
                axes[group_idx, i * 2].set_ylabel(group_name, color='white', fontsize=12, rotation=0, 
                                                   labelpad=30, va='center')
            
            # Reconstruction
            axes[group_idx, i * 2 + 1].imshow(rec)
            axes[group_idx, i * 2 + 1].axis('off')
    
    # Column titles
    for i in range(n_samples):
        axes[0, i * 2].set_title('Original', color='white', fontsize=10)
        axes[0, i * 2 + 1].set_title('Recon', color='white', fontsize=10)
    
    fig.suptitle('VAE Reconstructions by Group', color='white', fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(gallery_dir / 'reconstruction_gallery.png', dpi=100, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {gallery_dir}")


@torch.no_grad() 
def generate_transition_matrix(model, vae, h5_dataset, device, output_dir):
    """Generate a matrix showing all 17x17 group transitions."""
    print("\n🔢 Generating transition matrix...")
    
    matrix_dir = output_dir / 'transition_matrix'
    matrix_dir.mkdir(exist_ok=True)
    
    # For each pair, show source -> target
    n_groups = 17
    
    fig, axes = plt.subplots(n_groups, n_groups, figsize=(40, 40))
    fig.patch.set_facecolor('#0a0a0a')
    
    for source_idx in tqdm(range(n_groups), desc="Building matrix"):
        source_name = IDX_TO_GROUP[source_idx]
        source_indices = h5_dataset.indices_by_group.get(source_idx, [])
        
        if len(source_indices) == 0:
            continue
        
        # Get source pattern
        pattern = h5_dataset.get_pattern_by_group(source_idx, 0).unsqueeze(0).to(device)
        with autocast('cuda'):
            z_start = vae.encode(pattern, source_name)
        
        for target_idx in range(n_groups):
            target_name = IDX_TO_GROUP[target_idx]
            
            try:
                src_t = torch.tensor([source_idx], device=device)
                tgt_t = torch.tensor([target_idx], device=device)
                
                # Get endpoint
                z_end = model.sample_endpoint(z_start, src_t, tgt_t, n_steps=20)
                
                with autocast('cuda'):
                    img = vae.decode(z_end.float(), target_name)
                
                img_np = img[0].cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
                
                axes[source_idx, target_idx].imshow(img_np)
                axes[source_idx, target_idx].axis('off')
                
                if source_idx == target_idx:
                    # Highlight diagonal
                    for spine in axes[source_idx, target_idx].spines.values():
                        spine.set_visible(True)
                        spine.set_color('white')
                        spine.set_linewidth(3)
                        
            except Exception as e:
                axes[source_idx, target_idx].axis('off')
                continue
    
    # Labels
    for i in range(n_groups):
        axes[i, 0].set_ylabel(IDX_TO_GROUP[i], color='white', fontsize=8, rotation=0, 
                              labelpad=20, va='center')
        axes[0, i].set_title(IDX_TO_GROUP[i], color='white', fontsize=8, rotation=45, ha='left')
    
    fig.suptitle('Transition Matrix: Source (rows) → Target (columns)', 
                 color='white', fontsize=24, fontweight='bold', y=0.92)
    
    plt.tight_layout()
    plt.savefig(matrix_dir / 'transition_matrix.png', dpi=80, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {matrix_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate Complete Visualizations')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Path to trained model directory (auto-detects latest if not specified)')
    parser.add_argument('--data-path', type=str, 
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5')
    args = parser.parse_args()
    
    # Auto-detect model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        output_base = Path('output')
        dirs = sorted(output_base.glob('flow_matching_*'), key=lambda x: x.stat().st_mtime, reverse=True)
        if dirs:
            model_dir = dirs[0]
            print(f"📂 Auto-detected: {model_dir}")
        else:
            print("❌ No trained models found. Train first with train_flow_matching.py")
            sys.exit(1)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Load models
    vae, model, vae_config = load_models(model_dir, device)
    
    # Load dataset
    print(f"\n📁 Loading dataset...")
    h5_dataset = H5PatternDataset(args.data_path, split='train')
    print(f"   {len(h5_dataset)} patterns")
    
    # Create output directory
    viz_output = model_dir / 'full_visualizations'
    viz_output.mkdir(exist_ok=True)
    
    # Generate all visualizations
    generate_all_transitions(model, vae, h5_dataset, device, viz_output, n_samples=2)
    generate_latent_space_visualization(model, vae, h5_dataset, device, viz_output)
    generate_reconstruction_gallery(vae, h5_dataset, device, viz_output)
    generate_transition_matrix(model, vae, h5_dataset, device, viz_output)
    
    print(f"\n✅ All visualizations saved to: {viz_output}")
    print("\nGenerated:")
    print("  📊 transitions/          - Transition strips and GIFs")
    print("  🎯 latent_space/         - Latent space with trajectories")
    print("  🖼️  reconstructions/      - VAE reconstruction gallery")
    print("  🔢 transition_matrix/    - 17x17 transition matrix")


if __name__ == '__main__':
    main()


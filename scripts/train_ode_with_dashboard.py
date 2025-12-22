#!/usr/bin/env python3
"""
Train Neural ODE for Phase Transitions with Real-Time Dashboard.

Optimized for RTX 4090 with mixed precision and parallel processing.

Usage:
    python scripts/train_ode_with_dashboard.py \
        --vae-checkpoint output/vae_rgb_256/best_model.pt \
        --epochs 100 \
        --dashboard-port 8080
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Use non-interactive backend to avoid conflicts with NiceGUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig
# Legacy import for backwards compatibility
try:
    from src.models.symmetry_invariant_vae_rgb import (
        SymmetryInvariantVAE_RGB,
        SymmetryVAEConfigRGB,
    )
except ImportError:
    pass
from src.models.neural_ode_transition import (
    NeuralODETransition,
    NeuralODEConfig,
    TransitionLoss,
    TransitionMetrics,
    ALL_17_GROUPS,
    GROUP_TO_IDX,
    IDX_TO_GROUP,
)
from src.dataset.transition_dataset import (
    H5PatternDataset,
    TransitionDataset,
)

# Dashboard imports (optional)
try:
    from scripts.dashboard.training_state import TrainingState
    from scripts.dashboard.dashboard import run_dashboard
    from scripts.dashboard.theme import GROUP_COLORS
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False
    print("Warning: Dashboard not available. Install nicegui: pip install nicegui")
    GROUP_COLORS = {g: '#888888' for g in ALL_17_GROUPS}


# Canonical transitions for visualization
CANONICAL_TRANSITIONS = [
    ('p1', 'p6m'),   # No symmetry → full hexagonal
    ('p1', 'p4m'),   # No symmetry → full square
    ('p2', 'p4'),    # 2-fold → 4-fold
    ('p3', 'p6'),    # 3-fold → 6-fold
    ('pm', 'pmm'),   # Add perpendicular reflection
    ('p4', 'p4m'),   # Add reflections to square
    ('p3', 'p3m1'),  # Add reflections to hexagonal
    ('cmm', 'p4m'),  # Rectangular → Square
]


def setup_cuda_optimizations():
    """Setup CUDA optimizations for RTX 4090."""
    if torch.cuda.is_available():
        # Enable cudnn benchmark
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        return True
    return False


def load_vae(checkpoint_path: Path, device: torch.device):
    """Load pretrained VAE (supports both SimpleVAE and legacy SymmetryInvariantVAE)."""
    print(f"Loading VAE from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    
    # Detect VAE type from config
    if config is None:
        # Default to SimpleVAE
        config = SimpleVAEConfig(latent_dim=64)
        vae = SimpleVAE(config)
    elif isinstance(config, SimpleVAEConfig) or hasattr(config, 'hidden_dims'):
        # SimpleVAE
        if not isinstance(config, SimpleVAEConfig):
            config = SimpleVAEConfig(latent_dim=config.latent_dim)
        vae = SimpleVAE(config)
    else:
        # Legacy SymmetryInvariantVAE
        vae = SymmetryInvariantVAE_RGB(config)
    
    vae = vae.to(device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    
    print(f"VAE loaded: latent_dim={config.latent_dim}")
    return vae, config


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, use_amp=True, state=None):
    """Train for one epoch with mixed precision."""
    model.train()
    
    total_losses = {
        'loss': 0.0,
        'endpoint_loss': 0.0,
        'smoothness_loss': 0.0,
        'velocity_loss': 0.0,
    }
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (z_source, z_target, source_idx, target_idx) in enumerate(pbar):
        z_source = z_source.to(device, non_blocking=True)
        z_target = z_target.to(device, non_blocking=True)
        source_idx = source_idx.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Simple forward pass - process entire batch at once
        with autocast(enabled=use_amp):
            trajectory = model(z_source, source_idx, target_idx, n_steps=10, return_trajectory=True)
            losses = loss_fn(trajectory, z_target)
            batch_loss = losses['loss']
        
        # Skip NaN batches
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Scaled backward pass
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Stricter clipping
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
        
        # Update dashboard state
        if state is not None:
            state.update_batch(batch_idx + 1, len(dataloader), batch_loss.item())
    
    # Average over epoch
    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    
    return total_losses


@torch.no_grad()
def visualize_latent_trajectories(model, vae, h5_dataset, device, save_path, epoch, use_amp=True):
    """Visualize latent space with sample trajectories."""
    model.eval()
    vae.eval()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Sample patterns from each group and plot their latents
    for group_idx, group_name in enumerate(ALL_17_GROUPS):
        try:
            patterns = []
            indices = h5_dataset.indices_by_group.get(group_idx, [])
            for i in range(min(50, len(indices))):
                pattern = h5_dataset.get_pattern_by_group(group_idx, i)
                patterns.append(pattern)
            
            if not patterns:
                continue
            
            patterns = torch.stack(patterns).to(device)
            with autocast(enabled=use_amp):
                z = vae.encode(patterns, group_name)
            z_np = z.float().cpu().numpy()
            
            color = GROUP_COLORS.get(group_name, '#888888')
            ax.scatter(z_np[:, 0], z_np[:, 1], c=color, label=group_name,
                      alpha=0.5, s=15)
        except Exception as e:
            print(f"Warning: Could not plot group {group_name}: {e}")
    
    # Draw sample trajectories
    for source, target in CANONICAL_TRANSITIONS[:4]:
        try:
            source_idx = GROUP_TO_IDX[source]
            pattern = h5_dataset.get_pattern_by_group(source_idx, 0).unsqueeze(0).to(device)
            
            with autocast(enabled=use_amp):
                z_start = vae.encode(pattern, source)
            
            src_tensor = torch.tensor([source_idx], device=device)
            tgt_tensor = torch.tensor([GROUP_TO_IDX[target]], device=device)
            trajectory = model(z_start.float(), src_tensor, tgt_tensor, n_steps=30)
            
            traj_np = trajectory[:, 0].cpu().numpy()
            
            ax.plot(traj_np[:, 0], traj_np[:, 1], 'w-', alpha=0.8, linewidth=2)
            ax.scatter(traj_np[0, 0], traj_np[0, 1], c='white', s=100, marker='o', zorder=10)
            ax.scatter(traj_np[-1, 0], traj_np[-1, 1], c='white', s=100, marker='*', zorder=10)
        except Exception as e:
            print(f"Warning: Could not draw trajectory {source}→{target}: {e}")
    
    ax.set_xlabel('z₁', fontsize=12, color='white')
    ax.set_ylabel('z₂', fontsize=12, color='white')
    ax.set_title(f'Latent Space with Trajectories (Epoch {epoch})', 
                 fontsize=14, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white',
              labelcolor='white', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


@torch.no_grad()
def visualize_transition(model, vae, h5_dataset, device, source, target, 
                         save_path_png, save_path_gif, n_frames=40, use_amp=True):
    """Visualize a single transition as static and animated."""
    model.eval()
    vae.eval()
    
    source_idx = GROUP_TO_IDX[source]
    target_idx = GROUP_TO_IDX[target]
    
    # Get source pattern
    pattern = h5_dataset.get_pattern_by_group(source_idx, 0).unsqueeze(0).to(device)
    
    with autocast(enabled=use_amp):
        z_start = vae.encode(pattern, source)
    
    # Get trajectory
    src_tensor = torch.tensor([source_idx], device=device)
    tgt_tensor = torch.tensor([target_idx], device=device)
    trajectory = model(z_start.float(), src_tensor, tgt_tensor, n_steps=n_frames)
    
    # Decode trajectory with blended symmetry
    images = []
    for i, z in enumerate(trajectory):
        t = i / (n_frames - 1)
        alpha = 0.5 * (1 - np.cos(np.pi * t))  # Smooth step
        
        with autocast(enabled=use_amp):
            img_source = vae.decode(z.unsqueeze(0), source)
            img_target = vae.decode(z.unsqueeze(0), target)
        img = (1 - alpha) * img_source.float() + alpha * img_target.float()
        
        images.append(img[0].cpu().permute(1, 2, 0).numpy())
    
    # Static visualization (8 frames)
    n_show = 8
    indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
    
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2.5))
    fig.patch.set_facecolor('#0a0a0a')
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(np.clip(images[idx], 0, 1))
        t = idx / (n_frames - 1)
        ax.set_title(f't={t:.2f}', fontsize=10, color='white')
        ax.axis('off')
    
    fig.suptitle(f'{source} → {target}', fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=100, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    # Animated GIF
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.axis('off')
    
    im = ax.imshow(np.clip(images[0], 0, 1))
    title = ax.set_title(f'{source} → {target}: t=0.00', fontsize=12, color='white')
    
    def update(frame):
        im.set_array(np.clip(images[frame], 0, 1))
        t = frame / (n_frames - 1)
        title.set_text(f'{source} → {target}: t={t:.2f}')
        return [im, title]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    writer = PillowWriter(fps=20)
    anim.save(save_path_gif, writer=writer, savefig_kwargs={'facecolor': '#0a0a0a'})
    plt.close()


def generate_epoch_visualizations(model, vae, h5_dataset, device, output_dir, epoch, 
                                  state=None, use_amp=True):
    """Generate all visualizations for an epoch."""
    viz_dir = output_dir / 'visualizations' / f'epoch_{epoch:03d}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    trans_dir = viz_dir / 'transitions'
    trans_dir.mkdir(exist_ok=True)
    
    if state:
        state.log(f"Generating visualizations for epoch {epoch}...")
    
    # Latent space with trajectories
    latent_path = viz_dir / 'latent_space.png'
    visualize_latent_trajectories(model, vae, h5_dataset, device, latent_path, epoch, use_amp)
    if state:
        state.set_visualization('latent', str(latent_path))
        state.log(f"  Saved latent space visualization")
    
    # Canonical transitions
    for source, target in CANONICAL_TRANSITIONS:
        png_path = trans_dir / f'{source}_to_{target}.png'
        gif_path = trans_dir / f'{source}_to_{target}.gif'
        
        try:
            visualize_transition(model, vae, h5_dataset, device, 
                               source, target, png_path, gif_path, use_amp=use_amp)
            if state:
                state.set_visualization(f'transition_{source}_to_{target}', str(gif_path))
        except Exception as e:
            print(f"Warning: Could not generate transition {source}→{target}: {e}")
    
    if state:
        state.log(f"  Generated {len(CANONICAL_TRANSITIONS)} transition visualizations")
    
    return viz_dir


def main():
    parser = argparse.ArgumentParser(description='Train Neural ODE with Dashboard (RTX 4090 Optimized)')
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to pretrained VAE checkpoint')
    parser.add_argument('--data-path', type=str,
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5',
                       help='Path to H5 dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (128-256 for RTX 4090)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for ODE')
    parser.add_argument('--pairs-per-epoch', type=int, default=20000,
                       help='Number of transition pairs per epoch')
    parser.add_argument('--dashboard-port', type=int, default=8080,
                       help='Dashboard port (0 to disable)')
    parser.add_argument('--viz-interval', type=int, default=5,
                       help='Visualization interval (epochs)')
    parser.add_argument('--lambda-smooth', type=float, default=0.1,
                       help='Smoothness loss weight')
    parser.add_argument('--lambda-velocity', type=float, default=0.01,
                       help='Velocity loss weight')
    parser.add_argument('--num-workers', type=int, default=24,
                       help='Number of workers for data loading')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Setup CUDA optimizations
    has_cuda = setup_cuda_optimizations()
    device = torch.device('cuda' if has_cuda else 'cpu')
    print(f"Using device: {device}")
    
    use_amp = has_cuda and not args.no_amp
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/neural_ode_transitions_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize training state
    state = TrainingState() if HAS_DASHBOARD else None
    if state:
        state.output_dir = output_dir
        state.total_epochs = args.epochs
    
    # Start dashboard in background thread
    if HAS_DASHBOARD and args.dashboard_port > 0:
        print(f"\nStarting dashboard on port {args.dashboard_port}...")
        print(f"Open http://localhost:{args.dashboard_port} in your browser\n")
        
        def run_dashboard_thread():
            from nicegui import ui
            from scripts.dashboard.dashboard import TrainingDashboard
            
            dashboard = TrainingDashboard(state, output_dir)
            
            @ui.page('/')
            def main_page():
                dashboard.setup()
            
            ui.run(port=args.dashboard_port, title='Neural ODE Training', 
                   dark=True, reload=False, show=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard_thread, daemon=True)
        dashboard_thread.start()
        time.sleep(2)  # Give dashboard time to start
    
    # Load VAE
    vae, vae_config = load_vae(Path(args.vae_checkpoint), device)
    print(f"VAE loaded (latent_dim={vae_config.latent_dim})")
    
    if state:
        state.log(f"Loaded VAE from {args.vae_checkpoint}")
    
    # Load dataset
    print("\nLoading dataset...")
    h5_dataset = H5PatternDataset(args.data_path, split='train')
    print(f"Dataset: {len(h5_dataset)} patterns")
    
    if state:
        state.log(f"Loaded dataset with {len(h5_dataset)} patterns")
    
    # Create transition dataset
    print("\nCreating transition pairs...")
    transition_dataset = TransitionDataset(
        vae=vae,
        h5_dataset=h5_dataset,
        device=device,
        pairs_per_epoch=args.pairs_per_epoch,
        seed=42,
    )
    
    # Use appropriate number of workers
    num_workers = min(args.num_workers, 24)
    
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = 4
    
    dataloader = DataLoader(transition_dataset, **dataloader_kwargs)
    
    if state:
        state.log(f"Created {len(transition_dataset)} transition pairs")
    
    # Create Neural ODE model
    ode_config = NeuralODEConfig(
        latent_dim=vae_config.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_smooth=args.lambda_smooth,
        lambda_velocity=args.lambda_velocity,
    )
    
    model = NeuralODETransition(ode_config).to(device)
    
    # Optionally compile model
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Neural ODE parameters: {n_params:,}")
    
    if state:
        state.log(f"Created Neural ODE model ({n_params:,} parameters)")
    
    # Loss and optimizer
    loss_fn = TransitionLoss(ode_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'vae_checkpoint': str(args.vae_checkpoint),
            'latent_dim': ode_config.latent_dim,
            'hidden_dim': ode_config.hidden_dim,
            'lambda_smooth': ode_config.lambda_smooth,
            'lambda_velocity': ode_config.lambda_velocity,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'pairs_per_epoch': args.pairs_per_epoch,
            'use_amp': use_amp,
            'num_workers': num_workers,
        }, f, indent=2)
    
    # Training history
    history = {
        'loss': [], 'endpoint': [], 'smoothness': [], 'velocity': []
    }
    best_loss = float('inf')
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING NEURAL ODE (RTX 4090 Optimized)")
    print("="*60)
    
    if state:
        state.is_training = True
        state.start_time = time.time()
        state.log("Training started")
    
    train_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        if state:
            state.epoch = epoch + 1
            state.current_lr = scheduler.get_last_lr()[0]
        
        # Regenerate pairs each epoch for variety
        transition_dataset.regenerate_pairs()
        
        # Train epoch
        losses = train_epoch(model, dataloader, optimizer, loss_fn, device, 
                            scaler, use_amp, state)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['loss'].append(losses['loss'])
        history['endpoint'].append(losses['endpoint_loss'])
        history['smoothness'].append(losses['smoothness_loss'])
        history['velocity'].append(losses['velocity_loss'])
        
        if state:
            state.update_loss(epoch + 1, {
                'total': losses['loss'],
                'endpoint': losses['endpoint_loss'],
                'smoothness': losses['smoothness_loss'],
                'velocity': losses['velocity_loss'],
            })
            state.epoch_times.append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | "
              f"Loss: {losses['loss']:.6f} | "
              f"Endpoint: {losses['endpoint_loss']:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        if state:
            state.log(f"Epoch {epoch + 1} | Loss: {losses['loss']:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if losses['loss'] < best_loss:
            best_loss = losses['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': ode_config,
                'best_loss': best_loss,
            }, output_dir / 'best_model.pt')
            
            if state:
                state.log(f"  New best model saved (loss: {best_loss:.6f})")
        
        # Periodic visualizations
        if (epoch + 1) % args.viz_interval == 0:
            print(f"  Generating visualizations...")
            
            generate_epoch_visualizations(
                model, vae, h5_dataset, device, output_dir, epoch + 1, state, use_amp
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': ode_config,
            }, output_dir / f'checkpoint_epoch_{epoch + 1:03d}.pt')
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    total_time = time.time() - train_start
    
    # Final visualizations
    print("\nGenerating final visualizations...")
    generate_epoch_visualizations(model, vae, h5_dataset, device, output_dir, args.epochs, state, use_amp)
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': ode_config,
    }, output_dir / 'final_model.pt')
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0].plot(epochs, history['loss'], color='#4ecdc4', linewidth=2)
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Total Loss', color='white')
    axes[0].set_title('Total Loss', color='white', fontweight='bold')
    axes[0].set_yscale('log')
    
    axes[1].plot(epochs, history['endpoint'], label='Endpoint', color='#ff6b6b')
    axes[1].plot(epochs, history['smoothness'], label='Smoothness', color='#ffd93d')
    axes[1].plot(epochs, history['velocity'], label='Velocity', color='#a29bfe')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Loss', color='white')
    axes[1].set_title('Loss Components', color='white', fontweight='bold')
    axes[1].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    if state:
        state.is_training = False
        state.log("Training completed!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Average time per epoch: {total_time/args.epochs:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            print(f"  - {f.name}")
    
    print(f"\nVisualizations: {output_dir / 'visualizations'}")
    
    # Cleanup
    h5_dataset.close()
    
    # Keep running if dashboard is active
    if HAS_DASHBOARD and args.dashboard_port > 0:
        print(f"\nDashboard running at http://localhost:{args.dashboard_port}")
        print("Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()

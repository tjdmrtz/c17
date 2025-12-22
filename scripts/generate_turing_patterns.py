#!/usr/bin/env python3
"""
Generate Turing Patterns with Crystallographic Symmetries

This script generates beautiful visualizations of Turing patterns
(reaction-diffusion patterns) that respect the symmetries of the
17 wallpaper groups.

The key insight is that Turing patterns naturally form periodic structures,
and by constraining the simulation with symmetry projections, we can make
them follow any of the 17 crystallographic symmetry groups.

Usage:
    python scripts/generate_turing_patterns.py

Outputs:
    - output/turing_patterns/ - Individual patterns for each group
    - output/turing_patterns/gallery_all_17.png - Gallery of all groups
    - output/turing_patterns/pattern_comparison.png - Same group, different patterns
    - output/turing_patterns/blended_patterns.png - Blended/combined symmetries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import time

from src.dataset.turing_patterns import (
    TuringPatternGenerator, 
    TuringBlender,
    PatternType,
    TuringParams
)


# Custom colormaps for beautiful visualizations
def create_custom_colormaps():
    """Create beautiful custom colormaps for Turing patterns."""
    
    # Ocean depths
    ocean = LinearSegmentedColormap.from_list('ocean_depths', [
        '#0a1628', '#0d2137', '#103a5c', '#1a5276', '#2471a3',
        '#5499c7', '#85c1e9', '#aed6f1', '#d4e6f1', '#eaf2f8'
    ])
    
    # Forest moss
    forest = LinearSegmentedColormap.from_list('forest_moss', [
        '#1a1a0a', '#2d3319', '#3d4b24', '#4d6330', '#5e7d3a',
        '#7daa50', '#9ec46c', '#bfd98b', '#d9e8b0', '#f0f5d8'
    ])
    
    # Sunset coral
    coral = LinearSegmentedColormap.from_list('sunset_coral', [
        '#1a0a14', '#3d1c2a', '#6b2c42', '#9b3d5a', '#c94d6d',
        '#e07285', '#f099a8', '#f5bdc7', '#fad4dd', '#fef0f2'
    ])
    
    # Crystalline (cold blue-purple)
    crystal = LinearSegmentedColormap.from_list('crystalline', [
        '#0a0a1a', '#1a1a3d', '#2a2a5f', '#3d3d82', '#5050a5',
        '#6b6bc8', '#8585db', '#a5a5eb', '#c5c5f5', '#e8e8ff'
    ])
    
    # Golden amber
    amber = LinearSegmentedColormap.from_list('golden_amber', [
        '#1a0f00', '#3d2200', '#5f3500', '#824800', '#a55b00',
        '#c87000', '#db8c1a', '#eaaa40', '#f5c878', '#fff5e0'
    ])
    
    # Volcanic
    volcanic = LinearSegmentedColormap.from_list('volcanic', [
        '#0a0000', '#2d0a0a', '#4f1515', '#721f1f', '#942a2a',
        '#b53535', '#d64545', '#e86565', '#f59595', '#ffc8c8'
    ])
    
    return {
        'ocean': ocean,
        'forest': forest,
        'coral': coral,
        'crystal': crystal,
        'amber': amber,
        'volcanic': volcanic,
        'viridis': 'viridis',
        'plasma': 'plasma',
        'magma': 'magma',
        'cividis': 'cividis',
    }


def generate_all_17_gallery(output_dir: Path, resolution: int = 256, steps: int = 5000):
    """Generate a gallery showing Turing patterns for all 17 wallpaper groups."""
    
    print("="*70)
    print("GENERATING TURING PATTERNS FOR ALL 17 WALLPAPER GROUPS")
    print("="*70)
    
    gen = TuringPatternGenerator(resolution=resolution, seed=42)
    colormaps = create_custom_colormaps()
    
    # Create figure with all 17 groups (4x5 grid, leaving some empty)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Define groups with their lattice types for coloring
    groups_info = [
        # Oblique lattice
        ('p1', 'oblique', 'ocean'),
        ('p2', 'oblique', 'ocean'),
        # Rectangular lattice
        ('pm', 'rectangular', 'forest'),
        ('pg', 'rectangular', 'forest'),
        ('cm', 'rectangular', 'forest'),
        ('pmm', 'rectangular', 'coral'),
        ('pmg', 'rectangular', 'coral'),
        ('pgg', 'rectangular', 'coral'),
        ('cmm', 'rectangular', 'coral'),
        # Square lattice
        ('p4', 'square', 'crystal'),
        ('p4m', 'square', 'crystal'),
        ('p4g', 'square', 'crystal'),
        # Hexagonal lattice
        ('p3', 'hexagonal', 'amber'),
        ('p3m1', 'hexagonal', 'amber'),
        ('p31m', 'hexagonal', 'amber'),
        ('p6', 'hexagonal', 'volcanic'),
        ('p6m', 'hexagonal', 'volcanic'),
    ]
    
    for idx, (ax, (group, lattice, cmap_name)) in enumerate(zip(axes.flat, groups_info)):
        print(f"Generating {group} ({lattice})...")
        start = time.time()
        
        # Choose pattern type based on lattice
        if lattice == 'hexagonal':
            pattern_type = PatternType.SPOTS
        elif lattice == 'square':
            pattern_type = PatternType.SPOTS
        else:
            pattern_type = PatternType.STRIPES
        
        pattern = gen.generate(group, pattern_type, steps=steps)
        
        # Get colormap
        cmap = colormaps.get(cmap_name, 'viridis')
        
        ax.imshow(pattern, cmap=cmap, interpolation='lanczos')
        ax.set_title(f'{group}\n({lattice})', fontsize=12, color='white', 
                     fontweight='bold', pad=5)
        ax.axis('off')
        
        elapsed = time.time() - start
        print(f"  -> Completed in {elapsed:.1f}s")
    
    # Hide remaining axes
    for ax in axes.flat[len(groups_info):]:
        ax.axis('off')
    
    plt.suptitle('Turing Patterns × 17 Wallpaper Groups\n'
                 'Reaction-Diffusion with Crystallographic Symmetries',
                 fontsize=18, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = output_dir / 'gallery_all_17.png'
    plt.savefig(output_path, dpi=200, facecolor='#0a0a0a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved gallery to {output_path}")
    return output_path


def generate_pattern_type_comparison(output_dir: Path, resolution: int = 256, steps: int = 5000):
    """Compare different pattern types for selected wallpaper groups."""
    
    print("\n" + "="*70)
    print("COMPARING PATTERN TYPES FOR SELECTED GROUPS")
    print("="*70)
    
    colormaps = create_custom_colormaps()
    
    # Select groups to show
    groups = ['p4m', 'p6m', 'pmm', 'cm']
    pattern_types = [PatternType.SPOTS, PatternType.STRIPES, PatternType.MAZE, PatternType.CORAL]
    
    fig, axes = plt.subplots(len(groups), len(pattern_types), figsize=(16, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    for i, group in enumerate(groups):
        gen = TuringPatternGenerator(resolution=resolution, seed=42 + i)
        
        for j, ptype in enumerate(pattern_types):
            print(f"Generating {group} with {ptype.value}...")
            
            pattern = gen.generate(group, ptype, steps=steps)
            
            # Alternate colormaps
            cmap = list(colormaps.values())[((i + j) % len(colormaps))]
            
            axes[i, j].imshow(pattern, cmap=cmap, interpolation='lanczos')
            axes[i, j].axis('off')
            
            if i == 0:
                axes[i, j].set_title(ptype.value, fontsize=14, color='white',
                                     fontweight='bold', pad=10)
            if j == 0:
                axes[i, j].set_ylabel(group, fontsize=14, color='white',
                                      fontweight='bold', rotation=0, 
                                      labelpad=30, va='center')
    
    plt.suptitle('Pattern Types × Wallpaper Groups\n'
                 'Same Symmetry, Different Reaction-Diffusion Parameters',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    output_path = output_dir / 'pattern_comparison.png'
    plt.savefig(output_path, dpi=200, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison to {output_path}")
    return output_path


def generate_blended_patterns(output_dir: Path, resolution: int = 256, steps: int = 4000):
    """Generate patterns that blend or combine multiple symmetries."""
    
    print("\n" + "="*70)
    print("GENERATING BLENDED SYMMETRY PATTERNS")
    print("="*70)
    
    blender = TuringBlender(resolution=resolution, seed=123)
    colormaps = create_custom_colormaps()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    # 1. Gradient from p4m to p6m
    print("Creating p4m → p6m gradient...")
    gradient = blender.create_symmetry_gradient('p4m', 'p6m', PatternType.SPOTS, steps=steps)
    axes[0, 0].imshow(gradient, cmap=colormaps['crystal'], interpolation='lanczos')
    axes[0, 0].set_title('p4m → p6m Gradient', fontsize=12, color='white', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Gradient from pm to pgg
    print("Creating pm → pgg gradient...")
    gradient2 = blender.create_symmetry_gradient('pm', 'pgg', PatternType.STRIPES, steps=steps)
    axes[0, 1].imshow(gradient2, cmap=colormaps['forest'], interpolation='lanczos')
    axes[0, 1].set_title('pm → pgg Gradient', fontsize=12, color='white', fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Mosaic of 4 groups
    print("Creating 2×2 symmetry mosaic...")
    mosaic = blender.create_domain_mosaic(['p4m', 'p6m', 'pmm', 'p3m1'], 
                                           grid_size=2, 
                                           pattern_type=PatternType.SPOTS,
                                           steps=steps)
    axes[0, 2].imshow(mosaic, cmap=colormaps['coral'], interpolation='lanczos')
    axes[0, 2].set_title('4-Group Mosaic', fontsize=12, color='white', fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. Superimposed patterns (stripes + spots)
    print("Creating superimposed patterns...")
    superimposed = blender.superimpose_patterns('pmm', 'p6m', 
                                                 PatternType.STRIPES, PatternType.SPOTS,
                                                 overlay_scale=0.4, steps=steps)
    axes[1, 0].imshow(superimposed, cmap=colormaps['amber'], interpolation='lanczos')
    axes[1, 0].set_title('pmm Stripes + p6m Spots', fontsize=12, color='white', fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. Blended p4 and p6 (intersection of symmetries)
    print("Creating blended p4 + p6 patterns...")
    gen = TuringPatternGenerator(resolution, seed=42)
    p4_pattern = gen.generate('p4', PatternType.MAZE, steps=steps)
    p6_pattern = gen.generate('p6', PatternType.MAZE, steps=steps)
    blended = blender.blend_patterns([p4_pattern, p6_pattern], mode='average')
    axes[1, 1].imshow(blended, cmap=colormaps['volcanic'], interpolation='lanczos')
    axes[1, 1].set_title('p4 + p6 Average (→ p2)', fontsize=12, color='white', fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. Multi-scale pattern
    print("Creating multi-scale pattern...")
    multi = blender.superimpose_patterns('p6m', 'p4m',
                                          PatternType.SPOTS, PatternType.SPOTS,
                                          overlay_scale=0.25, steps=steps)
    axes[1, 2].imshow(multi, cmap=colormaps['ocean'], interpolation='lanczos')
    axes[1, 2].set_title('Multi-Scale (p6m large + p4m small)', fontsize=12, 
                         color='white', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Blended Turing Patterns\n'
                 'Combining Multiple Crystallographic Symmetries',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    output_path = output_dir / 'blended_patterns.png'
    plt.savefig(output_path, dpi=200, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved blended patterns to {output_path}")
    return output_path


def generate_individual_patterns(output_dir: Path, resolution: int = 512, steps: int = 8000):
    """Generate high-resolution individual patterns for each group."""
    
    print("\n" + "="*70)
    print("GENERATING HIGH-RES INDIVIDUAL PATTERNS")
    print("="*70)
    
    gen = TuringPatternGenerator(resolution=resolution, seed=42)
    colormaps = create_custom_colormaps()
    cmap_list = list(colormaps.values())
    
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(exist_ok=True)
    
    for i, group in enumerate(TuringPatternGenerator.ALL_GROUPS):
        print(f"Generating high-res {group}...")
        
        # Choose appropriate pattern type
        if group in ['p3', 'p3m1', 'p31m', 'p6', 'p6m']:
            ptype = PatternType.SPOTS
        elif group in ['p4', 'p4m', 'p4g']:
            ptype = PatternType.SPOTS
        else:
            ptype = PatternType.STRIPES
        
        pattern = gen.generate(group, ptype, steps=steps)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('#0a0a0a')
        
        cmap = cmap_list[i % len(cmap_list)]
        ax.imshow(pattern, cmap=cmap, interpolation='lanczos')
        ax.axis('off')
        ax.set_title(f'Turing Pattern with {group} Symmetry', 
                     fontsize=14, color='white', fontweight='bold', pad=10)
        
        output_path = individual_dir / f'turing_{group}.png'
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a',
                    edgecolor='none', bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    return individual_dir


def generate_evolution_animation_frames(output_dir: Path, group: str = 'p6m', 
                                        resolution: int = 256, 
                                        total_steps: int = 5000,
                                        save_every: int = 100):
    """Generate frames showing the evolution of a Turing pattern."""
    
    print(f"\n" + "="*70)
    print(f"GENERATING EVOLUTION FRAMES FOR {group}")
    print("="*70)
    
    from src.dataset.turing_patterns import SymmetryProjector, TuringParams
    from scipy.ndimage import convolve
    
    frames_dir = output_dir / 'evolution_frames' / group
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    colormaps = create_custom_colormaps()
    
    # Initialize
    n = resolution
    rng = np.random.default_rng(42)
    projector = SymmetryProjector(resolution)
    params = TuringParams.for_pattern(PatternType.SPOTS)
    
    laplacian = np.array([
        [0.05, 0.2, 0.05],
        [0.2, -1.0, 0.2],
        [0.05, 0.2, 0.05]
    ])
    
    # Initialize concentrations
    u = np.ones((n, n))
    v = np.zeros((n, n))
    
    noise_u = rng.uniform(-0.1, 0.1, (n, n))
    noise_v = rng.uniform(0, 0.1, (n, n))
    
    noise_u = projector.project(noise_u, group)
    noise_v = projector.project(noise_v, group)
    
    # Add seed
    cx, cy = n // 2, n // 2
    for _ in range(5):
        bx = rng.integers(n // 4, 3 * n // 4)
        by = rng.integers(n // 4, 3 * n // 4)
        radius = rng.integers(n // 20, n // 8)
        y, x = np.ogrid[:n, :n]
        mask = ((x - bx)**2 + (y - by)**2) < radius**2
        v[mask] = 0.25
    
    v = projector.project(v, group)
    
    u += noise_u
    v += noise_v
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    frame_idx = 0
    
    for step in range(total_steps):
        # Gray-Scott step
        lap_u = convolve(u, laplacian, mode='wrap')
        lap_v = convolve(v, laplacian, mode='wrap')
        
        uvv = u * v * v
        
        du = params.Du * lap_u - uvv + params.f * (1 - u)
        dv = params.Dv * lap_v + uvv - (params.f + params.k) * v
        
        u = np.clip(u + params.dt * du, 0, 1)
        v = np.clip(v + params.dt * dv, 0, 1)
        
        # Symmetry projection
        if (step + 1) % 100 == 0:
            u = projector.project(u, group)
            v = projector.project(v, group)
        
        # Save frame
        if step % save_every == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor('#0a0a0a')
            
            ax.imshow(v, cmap=colormaps['crystal'], interpolation='lanczos', 
                      vmin=0, vmax=0.5)
            ax.axis('off')
            ax.set_title(f'{group} Evolution - Step {step}', 
                         fontsize=12, color='white', fontweight='bold')
            
            frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
            plt.savefig(frame_path, dpi=100, facecolor='#0a0a0a',
                        edgecolor='none', bbox_inches='tight')
            plt.close()
            
            frame_idx += 1
            
            if step % 1000 == 0:
                print(f"  Step {step}/{total_steps} - Frame {frame_idx}")
    
    print(f"\nSaved {frame_idx} frames to {frames_dir}")
    print("To create video: ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 evolution.mp4")
    
    return frames_dir


def main():
    """Main function to generate all visualizations."""
    
    # Setup output directory
    output_dir = Path("output/turing_patterns")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("TURING PATTERNS × CRYSTALLOGRAPHIC SYMMETRIES")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nThis script generates Turing patterns (reaction-diffusion)")
    print("that respect the symmetries of the 17 wallpaper groups.")
    print()
    
    # Configuration
    resolution = 256
    steps = 5000
    
    # Generate visualizations
    generate_all_17_gallery(output_dir, resolution=resolution, steps=steps)
    generate_pattern_type_comparison(output_dir, resolution=resolution, steps=steps)
    generate_blended_patterns(output_dir, resolution=resolution, steps=4000)
    
    # High-res individual patterns (optional - takes longer)
    # generate_individual_patterns(output_dir, resolution=512, steps=8000)
    
    # Evolution frames (optional - for animation)
    # generate_evolution_animation_frames(output_dir, 'p6m', resolution=256, 
    #                                      total_steps=5000, save_every=50)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in: {output_dir.absolute()}")
    print("\nFiles:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()


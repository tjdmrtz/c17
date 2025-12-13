#!/usr/bin/env python3
"""
Script to generate and visualize beautiful 3D crystallographic patterns.

This script demonstrates both:
1. The 17 wallpaper groups extended to 3D
2. The 230 space groups (3D crystallographic groups)

Output includes volumetric patterns and beautiful visualizations.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.pattern_generator_3d import (
    WallpaperGroup3DGenerator, 
    ExtrusionType,
    visualize_3d_pattern,
    create_slices_visualization
)
from src.dataset.space_group_generator import (
    SpaceGroupGenerator,
    CrystalSystem,
    SPACE_GROUPS,
    visualize_space_group
)


def create_beautiful_visualization(pattern: np.ndarray, 
                                   output_path: str,
                                   title: str = "",
                                   colormap: str = "magma"):
    """Create a beautiful multi-view visualization of a 3D pattern."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom colormap for more dramatic effect
    colors_dramatic = [
        (0.0, '#0d0221'),    # Deep purple-black
        (0.2, '#3d1a78'),    # Dark purple
        (0.4, '#6b2d5c'),    # Magenta
        (0.6, '#a83279'),    # Pink
        (0.8, '#e8567c'),    # Coral
        (1.0, '#ffb997'),    # Peach
    ]
    custom_cmap = LinearSegmentedColormap.from_list('crystalline', 
        [c[1] for c in colors_dramatic])
    
    fig = plt.figure(figsize=(16, 10), facecolor='#0d0221')
    
    sz, sy, sx = pattern.shape
    
    # Central cross-section slices
    ax1 = fig.add_subplot(231, facecolor='#0d0221')
    ax1.imshow(pattern[sz//2], cmap=custom_cmap, vmin=0, vmax=1)
    ax1.set_title('XY Plane (Z center)', color='white', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(232, facecolor='#0d0221')
    ax2.imshow(pattern[:, sy//2, :], cmap=custom_cmap, vmin=0, vmax=1)
    ax2.set_title('XZ Plane (Y center)', color='white', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(233, facecolor='#0d0221')
    ax3.imshow(pattern[:, :, sx//2], cmap=custom_cmap, vmin=0, vmax=1)
    ax3.set_title('YZ Plane (X center)', color='white', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Maximum intensity projection
    ax4 = fig.add_subplot(234, facecolor='#0d0221')
    mip_z = np.max(pattern, axis=0)
    ax4.imshow(mip_z, cmap=custom_cmap, vmin=0, vmax=1)
    ax4.set_title('Max Intensity (Z projection)', color='white', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Average projection
    ax5 = fig.add_subplot(235, facecolor='#0d0221')
    avg_z = np.mean(pattern, axis=0)
    ax5.imshow(avg_z, cmap=custom_cmap, vmin=0, vmax=avg_z.max())
    ax5.set_title('Average (Z projection)', color='white', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Depth-colored composite
    ax6 = fig.add_subplot(236, facecolor='#0d0221')
    depth_composite = np.zeros((sy, sx, 3))
    for z in range(sz):
        t = z / sz
        # Color varies with depth
        r = 0.5 + 0.5 * np.sin(2 * np.pi * t)
        g = 0.5 + 0.5 * np.sin(2 * np.pi * (t + 1/3))
        b = 0.5 + 0.5 * np.sin(2 * np.pi * (t + 2/3))
        depth_composite[:, :, 0] += pattern[z] * r
        depth_composite[:, :, 1] += pattern[z] * g
        depth_composite[:, :, 2] += pattern[z] * b
    depth_composite = depth_composite / depth_composite.max()
    ax6.imshow(depth_composite)
    ax6.set_title('Depth-Colored Composite', color='white', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Title
    fig.suptitle(title, color='white', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d0221')
    plt.close()


def generate_wallpaper_3d_gallery(output_dir: Path, resolution: int = 64, seed: int = 42):
    """Generate a gallery of all 17 wallpaper groups in 3D."""
    print("\n" + "="*60)
    print("Generating 3D Wallpaper Groups Gallery")
    print("="*60)
    
    gallery_dir = output_dir / "wallpaper_3d"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    generator = WallpaperGroup3DGenerator(
        resolution=(resolution, resolution, resolution),
        seed=seed
    )
    
    # Generate with different extrusion types for variety
    extrusions = {
        "p1": ExtrusionType.WAVE,
        "p2": ExtrusionType.HELIX,
        "pm": ExtrusionType.WAVE,
        "pg": ExtrusionType.WAVE,
        "cm": ExtrusionType.WAVE,
        "pmm": ExtrusionType.CRYSTAL,
        "pmg": ExtrusionType.WAVE,
        "pgg": ExtrusionType.WAVE,
        "cmm": ExtrusionType.CRYSTAL,
        "p4": ExtrusionType.HELIX,
        "p4m": ExtrusionType.CRYSTAL,
        "p4g": ExtrusionType.WAVE,
        "p3": ExtrusionType.HELIX,
        "p3m1": ExtrusionType.WAVE,
        "p31m": ExtrusionType.WAVE,
        "p6": ExtrusionType.HELIX,
        "p6m": ExtrusionType.CRYSTAL,
    }
    
    for group_name in generator.list_groups():
        print(f"  Generating {group_name}...", end=" ", flush=True)
        
        extrusion = extrusions.get(group_name, ExtrusionType.WAVE)
        pattern = generator.generate(
            group_name, 
            motif_size=resolution//4,
            extrusion=extrusion,
            complexity=4,
            motif_type="mixed"
        )
        
        # Save visualization
        output_path = gallery_dir / f"{group_name}_3d.png"
        create_beautiful_visualization(
            pattern, 
            str(output_path),
            title=f"3D Wallpaper Group: {group_name}"
        )
        
        # Also save the raw data
        np.save(gallery_dir / f"{group_name}_3d.npy", pattern)
        
        print(f"✓ saved to {output_path.name}")
    
    print(f"\nWallpaper 3D gallery saved to: {gallery_dir}")
    return gallery_dir


def generate_space_groups_gallery(output_dir: Path, 
                                  resolution: int = 64, 
                                  seed: int = 42,
                                  samples_per_system: int = 3):
    """Generate a gallery of space groups from each crystal system."""
    print("\n" + "="*60)
    print("Generating 230 Space Groups Gallery")
    print("="*60)
    
    gallery_dir = output_dir / "space_groups"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SpaceGroupGenerator(
        resolution=(resolution, resolution, resolution),
        seed=seed
    )
    
    # Generate samples from each crystal system
    for system in CrystalSystem:
        system_dir = gallery_dir / system.value
        system_dir.mkdir(exist_ok=True)
        
        print(f"\n{system.value.upper()} System:")
        
        groups = generator.list_by_system(system)
        
        # Select representative groups (first, middle, last, and some random)
        rng = np.random.default_rng(seed)
        selected = list(set([
            groups[0], 
            groups[len(groups)//2], 
            groups[-1]
        ] + list(rng.choice(groups, size=min(samples_per_system, len(groups)), replace=False))))
        selected = sorted(selected)
        
        for group_num in selected:
            info = SPACE_GROUPS[group_num]
            print(f"  Generating #{group_num} ({info.symbol})...", end=" ", flush=True)
            
            pattern = generator.generate(
                group_num,
                motif_size=resolution//4,
                style="crystalline",
                num_atoms=4
            )
            
            # Save visualization
            safe_symbol = info.symbol.replace("/", "-").replace("₁", "1").replace("₂", "2").replace("₃", "3")
            output_path = system_dir / f"{group_num:03d}_{safe_symbol}.png"
            create_beautiful_visualization(
                pattern,
                str(output_path),
                title=f"Space Group #{group_num}: {info.symbol}\n{info.description}"
            )
            
            # Save raw data
            np.save(system_dir / f"{group_num:03d}_{safe_symbol}.npy", pattern)
            
            print(f"✓")
    
    print(f"\nSpace groups gallery saved to: {gallery_dir}")
    return gallery_dir


def generate_famous_structures(output_dir: Path, resolution: int = 64, seed: int = 42):
    """Generate famous crystal structures."""
    print("\n" + "="*60)
    print("Generating Famous Crystal Structures")
    print("="*60)
    
    structures_dir = output_dir / "famous_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SpaceGroupGenerator(
        resolution=(resolution, resolution, resolution),
        seed=seed
    )
    
    patterns = generator.generate_famous_structures(
        motif_size=resolution//4,
        style="molecular",
        num_atoms=4
    )
    
    for name, pattern in patterns.items():
        print(f"  Generating {name}...", end=" ", flush=True)
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        output_path = structures_dir / f"{safe_name}.png"
        
        create_beautiful_visualization(
            pattern,
            str(output_path),
            title=f"Crystal Structure: {name}"
        )
        
        np.save(structures_dir / f"{safe_name}.npy", pattern)
        print(f"✓")
    
    print(f"\nFamous structures saved to: {structures_dir}")
    return structures_dir


def generate_comparison_grid(output_dir: Path, resolution: int = 48, seed: int = 42):
    """Generate a comparison grid showing both 17 wallpaper groups and selected space groups."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    print("\n" + "="*60)
    print("Generating Comparison Grid")
    print("="*60)
    
    # Custom colormap
    colors = ['#0d0221', '#3d1a78', '#6b2d5c', '#a83279', '#e8567c', '#ffb997']
    custom_cmap = LinearSegmentedColormap.from_list('crystalline', colors)
    
    # Generate wallpaper patterns
    wp_gen = WallpaperGroup3DGenerator(resolution=(resolution, resolution, resolution), seed=seed)
    wallpaper_patterns = {}
    
    for group_name in wp_gen.list_groups():
        pattern = wp_gen.generate(group_name, motif_size=resolution//4, complexity=3)
        wallpaper_patterns[group_name] = pattern
    
    # Create wallpaper grid
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), facecolor='#0d0221')
    axes = axes.flatten()
    
    groups = list(wallpaper_patterns.keys())
    for idx, (ax, name) in enumerate(zip(axes[:17], groups)):
        pattern = wallpaper_patterns[name]
        # Show max intensity projection
        mip = np.max(pattern, axis=0)
        ax.imshow(mip, cmap=custom_cmap, vmin=0, vmax=1)
        ax.set_title(name, color='white', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplot
    axes[17].axis('off')
    
    fig.suptitle('The 17 Wallpaper Groups in 3D', 
                 color='white', fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / "wallpaper_3d_grid.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d0221')
    plt.close()
    print(f"  Wallpaper grid saved to: {output_path}")
    
    # Generate space group grid (one from each system)
    sg_gen = SpaceGroupGenerator(resolution=(resolution, resolution, resolution), seed=seed)
    
    representative_groups = {
        'Triclinic': 2,      # P-1
        'Monoclinic': 14,    # P2₁/c
        'Orthorhombic': 62,  # Pnma
        'Tetragonal': 139,   # I4/mmm
        'Trigonal': 166,     # R-3m
        'Hexagonal': 194,    # P6₃/mmc
        'Cubic': 225,        # Fm-3m
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='#0d0221')
    axes = axes.flatten()
    
    for idx, (system_name, group_num) in enumerate(representative_groups.items()):
        pattern = sg_gen.generate(group_num, motif_size=resolution//4, num_atoms=3)
        info = SPACE_GROUPS[group_num]
        
        mip = np.max(pattern, axis=0)
        axes[idx].imshow(mip, cmap=custom_cmap, vmin=0, vmax=1)
        axes[idx].set_title(f"{system_name}\n{info.symbol}", 
                           color='white', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    axes[7].axis('off')  # Hide last subplot
    
    fig.suptitle('The 7 Crystal Systems - Representative Space Groups', 
                 color='white', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / "crystal_systems_grid.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d0221')
    plt.close()
    print(f"  Crystal systems grid saved to: {output_path}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate beautiful 3D crystallographic patterns'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/3d_patterns',
        help='Output directory for generated patterns'
    )
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=64,
        help='Resolution of 3D volumes (default: 64)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--wallpaper',
        action='store_true',
        help='Generate 17 wallpaper groups in 3D'
    )
    parser.add_argument(
        '--space-groups',
        action='store_true',
        help='Generate 230 space groups gallery'
    )
    parser.add_argument(
        '--famous',
        action='store_true',
        help='Generate famous crystal structures'
    )
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Generate comparison grids'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all pattern types'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "🔮" * 30)
    print("  3D CRYSTALLOGRAPHIC PATTERN GENERATOR")
    print("🔮" * 30)
    print(f"\nOutput directory: {output_dir}")
    print(f"Resolution: {args.resolution}³")
    print(f"Seed: {args.seed}")
    
    # If no specific option selected, generate a demo
    if not any([args.wallpaper, args.space_groups, args.famous, args.grid, args.all]):
        print("\nNo specific option selected. Generating demo patterns...")
        args.grid = True
        args.famous = True
    
    if args.all:
        args.wallpaper = True
        args.space_groups = True
        args.famous = True
        args.grid = True
    
    if args.wallpaper:
        generate_wallpaper_3d_gallery(output_dir, args.resolution, args.seed)
    
    if args.space_groups:
        generate_space_groups_gallery(output_dir, args.resolution, args.seed)
    
    if args.famous:
        generate_famous_structures(output_dir, args.resolution, args.seed)
    
    if args.grid:
        generate_comparison_grid(output_dir, min(args.resolution, 48), args.seed)
    
    print("\n" + "✨" * 30)
    print("  GENERATION COMPLETE!")
    print("✨" * 30)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()




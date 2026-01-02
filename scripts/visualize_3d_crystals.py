#!/usr/bin/env python3
"""
Beautiful 3D Volumetric Visualization of Crystallographic Patterns

This script creates stunning true 3D renderings of crystallographic patterns
using isosurfaces and volumetric rendering techniques.

Each image shows a single pattern from a dramatic 3D perspective.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors

from src.dataset.pattern_generator_3d import WallpaperGroup3DGenerator, ExtrusionType
from src.dataset.space_group_generator import SpaceGroupGenerator, CrystalSystem, SPACE_GROUPS


# Beautiful color palettes for crystals
CRYSTAL_PALETTES = {
    'amethyst': {
        'face': '#9B59B6',
        'edge': '#6C3483',
        'bg': '#1a0a2e',
        'light': '#D7BDE2'
    },
    'emerald': {
        'face': '#1ABC9C',
        'edge': '#0E6655',
        'bg': '#0a1f1a',
        'light': '#A3E4D7'
    },
    'ruby': {
        'face': '#E74C3C',
        'edge': '#922B21',
        'bg': '#1a0a0a',
        'light': '#F5B7B1'
    },
    'sapphire': {
        'face': '#3498DB',
        'edge': '#1A5276',
        'bg': '#0a1520',
        'light': '#AED6F1'
    },
    'gold': {
        'face': '#F39C12',
        'edge': '#9A7D0A',
        'bg': '#1a1508',
        'light': '#F9E79F'
    },
    'diamond': {
        'face': '#ECF0F1',
        'edge': '#ABB2B9',
        'bg': '#17202A',
        'light': '#FFFFFF'
    },
    'obsidian': {
        'face': '#2C3E50',
        'edge': '#1B2631',
        'bg': '#0a0f14',
        'light': '#5D6D7E'
    },
    'rose_quartz': {
        'face': '#F5B7B1',
        'edge': '#D98880',
        'bg': '#1a1015',
        'light': '#FADBD8'
    },
    'citrine': {
        'face': '#F4D03F',
        'edge': '#B7950B',
        'bg': '#1a1a08',
        'light': '#FCF3CF'
    },
    'aquamarine': {
        'face': '#76D7C4',
        'edge': '#45B39D',
        'bg': '#081a18',
        'light': '#D1F2EB'
    }
}


def extract_isosurface(volume: np.ndarray, 
                       level: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract isosurface from 3D volume using marching cubes.
    
    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
    """
    # Pad volume to ensure closed surfaces
    padded = np.pad(volume, 1, mode='constant', constant_values=0)
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            padded, level=level, spacing=(1, 1, 1)
        )
        # Adjust for padding
        verts = verts - 1
        return verts, faces, normals
    except Exception as e:
        print(f"Warning: Could not extract isosurface: {e}")
        return None, None, None


def render_crystal_3d(volume: np.ndarray,
                      output_path: str,
                      title: str = "",
                      palette: str = 'amethyst',
                      iso_level: float = 0.35,
                      view_angle: Tuple[float, float] = (25, 45),
                      figsize: Tuple[int, int] = (12, 12),
                      show_wireframe: bool = False):
    """
    Render a beautiful 3D visualization of a crystallographic pattern.
    
    Args:
        volume: 3D numpy array with the pattern
        output_path: Path to save the image
        title: Title for the plot
        palette: Color palette name
        iso_level: Isosurface level (0-1)
        view_angle: (elevation, azimuth) viewing angles
        figsize: Figure size
        show_wireframe: Whether to show wireframe edges
    """
    colors = CRYSTAL_PALETTES.get(palette, CRYSTAL_PALETTES['amethyst'])
    
    # Extract isosurface
    verts, faces, normals = extract_isosurface(volume, level=iso_level)
    
    if verts is None or len(verts) == 0:
        print(f"Warning: No isosurface found for {title}")
        return
    
    # Create figure with dark background
    fig = plt.figure(figsize=figsize, facecolor=colors['bg'])
    ax = fig.add_subplot(111, projection='3d', facecolor=colors['bg'])
    
    # Create mesh
    mesh = Poly3DCollection(verts[faces])
    
    # Calculate face colors based on normals for lighting effect
    face_normals = normals[faces].mean(axis=1)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
    
    # Light direction
    light_dir = np.array([1, 1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Calculate diffuse lighting
    diffuse = np.abs(np.dot(face_normals, light_dir))
    diffuse = 0.3 + 0.7 * diffuse  # Ambient + diffuse
    
    # Convert base color to RGB
    base_rgb = mcolors.to_rgb(colors['face'])
    light_rgb = mcolors.to_rgb(colors['light'])
    
    # Apply lighting to create gradient colors
    face_colors = np.zeros((len(faces), 4))
    for i, d in enumerate(diffuse):
        # Blend between base and light color based on lighting
        r = base_rgb[0] * (1 - d * 0.3) + light_rgb[0] * (d * 0.3)
        g = base_rgb[1] * (1 - d * 0.3) + light_rgb[1] * (d * 0.3)
        b = base_rgb[2] * (1 - d * 0.3) + light_rgb[2] * (d * 0.3)
        face_colors[i] = [r, g, b, 0.9]
    
    mesh.set_facecolor(face_colors)
    
    if show_wireframe:
        mesh.set_edgecolor(colors['edge'])
        mesh.set_linewidth(0.3)
    else:
        mesh.set_edgecolor('none')
    
    ax.add_collection3d(mesh)
    
    # Set axis limits
    sz, sy, sx = volume.shape
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_zlim(0, sz)
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Set aspect ratio
    ax.set_box_aspect([sx, sy, sz])
    
    # Add title
    if title:
        ax.set_title(title, color='white', fontsize=16, fontweight='bold', 
                    pad=20, fontfamily='sans-serif')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor=colors['bg'], edgecolor='none')
    plt.close()


def render_crystal_voxels(volume: np.ndarray,
                          output_path: str,
                          title: str = "",
                          palette: str = 'amethyst',
                          threshold: float = 0.3,
                          view_angle: Tuple[float, float] = (25, 45),
                          figsize: Tuple[int, int] = (12, 12),
                          alpha_scale: float = 0.8):
    """
    Render 3D pattern using voxels with transparency based on intensity.
    """
    colors = CRYSTAL_PALETTES.get(palette, CRYSTAL_PALETTES['amethyst'])
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=colors['bg'])
    ax = fig.add_subplot(111, projection='3d', facecolor=colors['bg'])
    
    # Create voxel mask
    voxels = volume > threshold
    
    if not voxels.any():
        print(f"Warning: No voxels above threshold for {title}")
        return
    
    # Create color array with alpha based on intensity
    sz, sy, sx = volume.shape
    facecolors = np.zeros(voxels.shape + (4,))
    
    base_rgb = mcolors.to_rgb(colors['face'])
    light_rgb = mcolors.to_rgb(colors['light'])
    
    # Calculate colors based on position and intensity for 3D depth effect
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                if voxels[z, y, x]:
                    # Intensity-based coloring
                    intensity = volume[z, y, x]
                    
                    # Depth-based shading (lighter on top)
                    depth_factor = z / sz
                    
                    # Blend colors
                    r = base_rgb[0] * (1 - depth_factor * 0.3) + light_rgb[0] * (depth_factor * 0.3)
                    g = base_rgb[1] * (1 - depth_factor * 0.3) + light_rgb[1] * (depth_factor * 0.3)
                    b = base_rgb[2] * (1 - depth_factor * 0.3) + light_rgb[2] * (depth_factor * 0.3)
                    
                    # Alpha based on intensity
                    alpha = min(1.0, intensity * alpha_scale)
                    
                    facecolors[z, y, x] = [r, g, b, alpha]
    
    # Draw voxels
    ax.voxels(voxels, facecolors=facecolors, edgecolor='none')
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Remove axes
    ax.set_axis_off()
    
    # Set aspect ratio
    ax.set_box_aspect([sx, sy, sz])
    
    # Add title
    if title:
        ax.set_title(title, color='white', fontsize=16, fontweight='bold',
                    pad=20, fontfamily='sans-serif')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=colors['bg'], edgecolor='none')
    plt.close()


def render_multi_angle(volume: np.ndarray,
                       output_path: str,
                       title: str = "",
                       palette: str = 'amethyst',
                       iso_level: float = 0.35,
                       figsize: Tuple[int, int] = (16, 16)):
    """
    Render the pattern from 4 different angles in a single image.
    """
    colors = CRYSTAL_PALETTES.get(palette, CRYSTAL_PALETTES['amethyst'])
    
    # Extract isosurface once
    verts, faces, normals = extract_isosurface(volume, level=iso_level)
    
    if verts is None:
        return
    
    # Four viewing angles
    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    
    fig = plt.figure(figsize=figsize, facecolor=colors['bg'])
    
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d', facecolor=colors['bg'])
        
        # Create mesh
        mesh = Poly3DCollection(verts[faces])
        
        # Calculate lighting
        face_normals = normals[faces].mean(axis=1)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
        
        light_dir = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim)), 0.5])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        diffuse = np.abs(np.dot(face_normals, light_dir))
        diffuse = 0.3 + 0.7 * diffuse
        
        base_rgb = mcolors.to_rgb(colors['face'])
        light_rgb = mcolors.to_rgb(colors['light'])
        
        face_colors = np.zeros((len(faces), 4))
        for i, d in enumerate(diffuse):
            r = base_rgb[0] * (1 - d * 0.3) + light_rgb[0] * (d * 0.3)
            g = base_rgb[1] * (1 - d * 0.3) + light_rgb[1] * (d * 0.3)
            b = base_rgb[2] * (1 - d * 0.3) + light_rgb[2] * (d * 0.3)
            face_colors[i] = [r, g, b, 0.9]
        
        mesh.set_facecolor(face_colors)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)
        
        sz, sy, sx = volume.shape
        ax.set_xlim(0, sx)
        ax.set_ylim(0, sy)
        ax.set_zlim(0, sz)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect([sx, sy, sz])
    
    if title:
        fig.suptitle(title, color='white', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=colors['bg'], edgecolor='none')
    plt.close()


def generate_wallpaper_3d_renders(output_dir: Path,
                                   resolution: int = 48,
                                   seed: int = 42):
    """Generate beautiful 3D renders for all 17 wallpaper groups."""
    print("\n" + "=" * 60)
    print("🔮 Generating 3D Wallpaper Group Renders")
    print("=" * 60)
    
    render_dir = output_dir / "wallpaper_3d"
    render_dir.mkdir(parents=True, exist_ok=True)
    
    generator = WallpaperGroup3DGenerator(
        resolution=(resolution, resolution, resolution),
        seed=seed
    )
    
    # Assign different palettes to different groups for variety
    group_palettes = {
        'p1': 'diamond',
        'p2': 'sapphire', 
        'pm': 'aquamarine',
        'pg': 'emerald',
        'cm': 'citrine',
        'pmm': 'ruby',
        'pmg': 'amethyst',
        'pgg': 'gold',
        'cmm': 'rose_quartz',
        'p4': 'sapphire',
        'p4m': 'diamond',
        'p4g': 'emerald',
        'p3': 'amethyst',
        'p3m1': 'ruby',
        'p31m': 'gold',
        'p6': 'aquamarine',
        'p6m': 'citrine',
    }
    
    for group_name in generator.list_groups():
        print(f"  Rendering {group_name}...", end=" ", flush=True)
        
        pattern = generator.generate(
            group_name,
            motif_size=resolution // 3,
            complexity=4,
            motif_type="spherical"
        )
        
        palette = group_palettes.get(group_name, 'amethyst')
        
        # Render isosurface
        output_path = render_dir / f"{group_name}_3d.png"
        render_crystal_3d(
            pattern,
            str(output_path),
            title=f"Wallpaper Group: {group_name}",
            palette=palette,
            iso_level=0.3,
            view_angle=(25, 45)
        )
        
        print(f"✓")
    
    print(f"\n✨ Wallpaper 3D renders saved to: {render_dir}")
    return render_dir


def generate_space_group_renders(output_dir: Path,
                                  resolution: int = 48,
                                  seed: int = 42):
    """Generate beautiful 3D renders for representative space groups."""
    print("\n" + "=" * 60)
    print("💎 Generating Space Group 3D Renders")
    print("=" * 60)
    
    render_dir = output_dir / "space_groups_3d"
    render_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SpaceGroupGenerator(
        resolution=(resolution, resolution, resolution),
        seed=seed
    )
    
    # Representative groups from each system with palettes
    representatives = [
        (1, 'P1', 'Triclinic - No symmetry', 'obsidian'),
        (2, 'P-1', 'Triclinic - Inversion', 'diamond'),
        (14, 'P2₁/c', 'Monoclinic - Most common', 'sapphire'),
        (19, 'P2₁2₁2₁', 'Orthorhombic - Chiral', 'emerald'),
        (62, 'Pnma', 'Orthorhombic - Common', 'ruby'),
        (139, 'I4/mmm', 'Tetragonal - High symmetry', 'gold'),
        (166, 'R-3m', 'Trigonal', 'amethyst'),
        (194, 'P6₃/mmc', 'Hexagonal - HCP metals', 'aquamarine'),
        (221, 'Pm-3m', 'Cubic - Simple', 'citrine'),
        (225, 'Fm-3m', 'Cubic - FCC metals', 'rose_quartz'),
        (227, 'Fd-3m', 'Cubic - Diamond structure', 'diamond'),
        (230, 'Ia-3d', 'Cubic - Garnet', 'ruby'),
    ]
    
    for group_num, symbol, description, palette in representatives:
        print(f"  Rendering #{group_num} ({symbol})...", end=" ", flush=True)
        
        pattern = generator.generate(
            group_num,
            motif_size=resolution // 3,
            style="crystalline",
            num_atoms=4
        )
        
        safe_symbol = symbol.replace("/", "-").replace("₁", "1").replace("₂", "2").replace("₃", "3")
        output_path = render_dir / f"{group_num:03d}_{safe_symbol}.png"
        
        render_crystal_3d(
            pattern,
            str(output_path),
            title=f"#{group_num} {symbol}\n{description}",
            palette=palette,
            iso_level=0.3,
            view_angle=(30, 45)
        )
        
        print(f"✓")
    
    print(f"\n✨ Space group 3D renders saved to: {render_dir}")
    return render_dir


def generate_famous_structure_renders(output_dir: Path,
                                       resolution: int = 48,
                                       seed: int = 42):
    """Generate beautiful 3D renders for famous crystal structures."""
    print("\n" + "=" * 60)
    print("💠 Generating Famous Crystal Structure Renders")
    print("=" * 60)
    
    render_dir = output_dir / "famous_3d"
    render_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SpaceGroupGenerator(
        resolution=(resolution, resolution, resolution),
        seed=seed
    )
    
    # Famous structures with their space groups and colors
    famous = [
        ("Diamond", 227, 'diamond', "Fd-3m - Carbon allotrope"),
        ("NaCl", 225, 'sapphire', "Fm-3m - Rock Salt"),
        ("CsCl", 221, 'gold', "Pm-3m - Cesium Chloride"),
        ("Perovskite", 221, 'citrine', "Pm-3m - CaTiO₃ structure"),
        ("Rutile", 136, 'ruby', "P4₂/mnm - TiO₂"),
        ("Wurtzite", 186, 'emerald', "P6₃mc - ZnS structure"),
        ("Fluorite", 225, 'amethyst', "Fm-3m - CaF₂"),
        ("Spinel", 227, 'rose_quartz', "Fd-3m - MgAl₂O₄"),
        ("Garnet", 230, 'ruby', "Ia-3d - Silicate minerals"),
        ("Ice_Ih", 194, 'aquamarine', "P6₃/mmc - Hexagonal ice"),
        ("Graphite", 194, 'obsidian', "P6₃/mmc - Carbon layers"),
        ("Quartz", 152, 'diamond', "P3₁21 - SiO₂"),
    ]
    
    for name, group_num, palette, description in famous:
        print(f"  Rendering {name}...", end=" ", flush=True)
        
        pattern = generator.generate(
            group_num,
            motif_size=resolution // 3,
            style="molecular",
            num_atoms=4
        )
        
        output_path = render_dir / f"{name}.png"
        
        render_crystal_3d(
            pattern,
            str(output_path),
            title=f"{name.replace('_', ' ')}\n{description}",
            palette=palette,
            iso_level=0.3,
            view_angle=(30, 45)
        )
        
        print(f"✓")
    
    print(f"\n✨ Famous structure 3D renders saved to: {render_dir}")
    return render_dir


def generate_rotating_views(output_dir: Path,
                            resolution: int = 48,
                            seed: int = 42):
    """Generate multiple viewing angles for select structures."""
    print("\n" + "=" * 60)
    print("🌀 Generating Multi-Angle Views")
    print("=" * 60)
    
    render_dir = output_dir / "multi_angle"
    render_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a few examples with 4-angle views
    wp_gen = WallpaperGroup3DGenerator(resolution=(resolution, resolution, resolution), seed=seed)
    sg_gen = SpaceGroupGenerator(resolution=(resolution, resolution, resolution), seed=seed)
    
    examples = [
        ('wallpaper', 'p6m', 'citrine'),
        ('wallpaper', 'p4m', 'sapphire'),
        ('space_group', 225, 'diamond'),
        ('space_group', 227, 'amethyst'),
    ]
    
    for pattern_type, name, palette in examples:
        if pattern_type == 'wallpaper':
            print(f"  Rendering {name} multi-angle...", end=" ", flush=True)
            pattern = wp_gen.generate(name, motif_size=resolution // 3, complexity=4)
            title = f"Wallpaper Group: {name}"
            filename = f"wallpaper_{name}_multiangle.png"
        else:
            print(f"  Rendering space group #{name} multi-angle...", end=" ", flush=True)
            pattern = sg_gen.generate(name, motif_size=resolution // 3, num_atoms=4)
            info = SPACE_GROUPS[name]
            title = f"Space Group #{name}: {info.symbol}"
            filename = f"spacegroup_{name}_multiangle.png"
        
        output_path = render_dir / filename
        render_multi_angle(
            pattern,
            str(output_path),
            title=title,
            palette=palette,
            iso_level=0.3
        )
        
        print(f"✓")
    
    print(f"\n✨ Multi-angle renders saved to: {render_dir}")
    return render_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate beautiful true 3D crystallographic visualizations'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/3d_renders',
        help='Output directory for rendered images'
    )
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=48,
        help='Resolution of 3D volumes (default: 48)'
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
        help='Generate 17 wallpaper group 3D renders'
    )
    parser.add_argument(
        '--space-groups',
        action='store_true',
        help='Generate space group 3D renders'
    )
    parser.add_argument(
        '--famous',
        action='store_true',
        help='Generate famous crystal structure renders'
    )
    parser.add_argument(
        '--multi-angle',
        action='store_true',
        help='Generate multi-angle views'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all render types'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "💎" * 30)
    print("  TRUE 3D CRYSTAL RENDERER")
    print("💎" * 30)
    print(f"\nOutput directory: {output_dir}")
    print(f"Resolution: {args.resolution}³")
    print(f"Seed: {args.seed}")
    
    # Default: generate wallpaper if nothing specified
    if not any([args.wallpaper, args.space_groups, args.famous, args.multi_angle, args.all]):
        print("\nNo option specified. Generating wallpaper 3D renders...")
        args.wallpaper = True
    
    if args.all:
        args.wallpaper = True
        args.space_groups = True
        args.famous = True
        args.multi_angle = True
    
    if args.wallpaper:
        generate_wallpaper_3d_renders(output_dir, args.resolution, args.seed)
    
    if args.space_groups:
        generate_space_group_renders(output_dir, args.resolution, args.seed)
    
    if args.famous:
        generate_famous_structure_renders(output_dir, args.resolution, args.seed)
    
    if args.multi_angle:
        generate_rotating_views(output_dir, args.resolution, args.seed)
    
    print("\n" + "✨" * 30)
    print("  RENDERING COMPLETE!")
    print("✨" * 30)
    print(f"\nAll renders saved to: {output_dir}")


if __name__ == "__main__":
    main()





#!/usr/bin/env python3
"""
Beautiful Crystal Structure Renderer

Creates stunning 3D visualizations of crystallographic structures with:
- Perfect spherical atoms at lattice positions
- Elegant cylindrical bonds connecting atoms
- Professional lighting and materials
- True crystallographic symmetry
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


# Gorgeous color schemes
THEMES = {
    'crystal_blue': {
        'bg': '#0a0a1a',
        'atoms': ['#4FC3F7', '#29B6F6', '#03A9F4', '#039BE5'],
        'bonds': '#1565C0',
        'glow': '#81D4FA',
    },
    'amethyst': {
        'bg': '#0d0015',
        'atoms': ['#CE93D8', '#BA68C8', '#AB47BC', '#9C27B0'],
        'bonds': '#6A1B9A',
        'glow': '#E1BEE7',
    },
    'emerald': {
        'bg': '#001a0d',
        'atoms': ['#81C784', '#66BB6A', '#4CAF50', '#43A047'],
        'bonds': '#1B5E20',
        'glow': '#C8E6C9',
    },
    'ruby': {
        'bg': '#1a0505',
        'atoms': ['#EF9A9A', '#E57373', '#EF5350', '#F44336'],
        'bonds': '#B71C1C',
        'glow': '#FFCDD2',
    },
    'gold': {
        'bg': '#1a1400',
        'atoms': ['#FFE082', '#FFD54F', '#FFCA28', '#FFC107'],
        'bonds': '#FF8F00',
        'glow': '#FFF8E1',
    },
    'diamond': {
        'bg': '#0a0a12',
        'atoms': ['#FFFFFF', '#ECEFF1', '#CFD8DC', '#B0BEC5'],
        'bonds': '#546E7A',
        'glow': '#FFFFFF',
    },
    'sapphire': {
        'bg': '#00051a',
        'atoms': ['#90CAF9', '#64B5F6', '#42A5F5', '#2196F3'],
        'bonds': '#0D47A1',
        'glow': '#BBDEFB',
    },
    'sunset': {
        'bg': '#1a0a05',
        'atoms': ['#FFCC80', '#FFB74D', '#FF9800', '#F57C00'],
        'bonds': '#E65100',
        'glow': '#FFE0B2',
    },
}


def create_sphere_points(center: np.ndarray, radius: float, 
                         resolution: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sphere surface points for plotting."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_cylinder_points(start: np.ndarray, end: np.ndarray, 
                           radius: float, resolution: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create cylinder surface points for bonds."""
    # Direction vector
    v = end - start
    length = np.linalg.norm(v)
    if length < 1e-6:
        return None, None, None
    v = v / length
    
    # Find perpendicular vectors
    if abs(v[0]) < 0.9:
        perp1 = np.cross(v, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(v, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(v, perp1)
    
    # Create cylinder
    theta = np.linspace(0, 2 * np.pi, resolution)
    z_line = np.linspace(0, length, 2)
    theta_grid, z_grid = np.meshgrid(theta, z_line)
    
    x = start[0] + z_grid * v[0] + radius * (np.cos(theta_grid) * perp1[0] + np.sin(theta_grid) * perp2[0])
    y = start[1] + z_grid * v[1] + radius * (np.cos(theta_grid) * perp1[1] + np.sin(theta_grid) * perp2[1])
    z = start[2] + z_grid * v[2] + radius * (np.cos(theta_grid) * perp1[2] + np.sin(theta_grid) * perp2[2])
    
    return x, y, z


class CrystalLattice:
    """Generates crystal lattice structures for different symmetry groups."""
    
    def __init__(self, size: Tuple[int, int, int] = (3, 3, 3)):
        self.size = size
    
    def _apply_symmetry(self, base_positions: List[np.ndarray], 
                        symmetry_ops: List[callable]) -> List[np.ndarray]:
        """Apply symmetry operations to generate all equivalent positions."""
        all_positions = []
        for pos in base_positions:
            for op in symmetry_ops:
                new_pos = op(pos)
                # Check if position is unique
                is_unique = True
                for existing in all_positions:
                    if np.allclose(new_pos % 1, existing % 1, atol=0.05):
                        is_unique = False
                        break
                if is_unique:
                    all_positions.append(new_pos % 1)
        return all_positions
    
    def generate_cubic_p(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Simple cubic (Pm-3m) - atoms at corners."""
        nx, ny, nz = self.size
        atoms = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    atoms.append(np.array([i, j, k], dtype=float))
        
        # Bonds between nearest neighbors
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.9 < dist < 1.1:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_cubic_f(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Face-centered cubic (Fm-3m) - FCC metals like gold, aluminum."""
        nx, ny, nz = self.size
        atoms = []
        
        # Corner and face-center positions
        base = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    for b in base:
                        pos = b + np.array([i, j, k])
                        if pos[0] <= nx and pos[1] <= ny and pos[2] <= nz:
                            atoms.append(pos)
        
        # Nearest neighbor bonds (distance = sqrt(2)/2 ≈ 0.707)
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.65 < dist < 0.8:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_cubic_i(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Body-centered cubic (Im-3m) - BCC metals like iron."""
        nx, ny, nz = self.size
        atoms = []
        
        base = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
        ]
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    for b in base:
                        pos = b + np.array([i, j, k])
                        if pos[0] <= nx and pos[1] <= ny and pos[2] <= nz:
                            atoms.append(pos)
        
        # Bonds (distance = sqrt(3)/2 ≈ 0.866)
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.8 < dist < 0.95:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_diamond(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Diamond cubic (Fd-3m) - Carbon diamond structure."""
        nx, ny, nz = self.size
        atoms = []
        
        # Diamond has 8 atoms per unit cell
        base = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.25, 0.25, 0.25]),
            np.array([0.75, 0.75, 0.25]),
            np.array([0.75, 0.25, 0.75]),
            np.array([0.25, 0.75, 0.75]),
        ]
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for b in base:
                        pos = b + np.array([i, j, k])
                        atoms.append(pos)
        
        # Tetrahedral bonds (distance = sqrt(3)/4 ≈ 0.433)
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.4 < dist < 0.5:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_hexagonal(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Hexagonal close-packed (P6₃/mmc)."""
        nx, ny, nz = self.size
        atoms = []
        
        # HCP has 2 atoms per unit cell
        c_over_a = np.sqrt(8/3)  # Ideal c/a ratio
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    # Layer A
                    x = i + 0.5 * j
                    y = j * np.sqrt(3) / 2
                    z = k * c_over_a
                    atoms.append(np.array([x, y, z]))
                    
                    # Layer B (offset)
                    if k < nz:
                        x2 = i + 0.5 * j + 1/3
                        y2 = j * np.sqrt(3) / 2 + np.sqrt(3) / 6
                        z2 = (k + 0.5) * c_over_a
                        atoms.append(np.array([x2, y2, z2]))
        
        # Bonds
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.9 < dist < 1.15:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_tetragonal(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Tetragonal (I4/mmm)."""
        nx, ny, nz = self.size
        atoms = []
        c_over_a = 1.5
        
        base = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5 * c_over_a]),
        ]
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    for b in base:
                        pos = np.array([i, j, k * c_over_a]) + b
                        if pos[2] <= nz * c_over_a + 0.01:
                            atoms.append(pos)
        
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.85 < dist < 1.1:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_trigonal(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Trigonal/Rhombohedral (R-3m)."""
        nx, ny, nz = self.size
        atoms = []
        
        # Rhombohedral angle
        alpha = np.radians(60)
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x = i + j * np.cos(alpha) + k * np.cos(alpha)
                    y = j * np.sin(alpha) + k * np.sin(alpha) * np.cos(np.radians(30))
                    z = k * np.sin(alpha) * np.sin(np.radians(30)) + k * 0.8
                    atoms.append(np.array([x, y, z]))
        
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.9 < dist < 1.2:
                    bonds.append((idx, idx2))
        
        return atoms, bonds
    
    def generate_orthorhombic(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Orthorhombic (Pnma)."""
        nx, ny, nz = self.size
        atoms = []
        
        # Different lattice parameters
        a, b, c = 1.0, 1.3, 1.6
        
        base = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for bp in base:
                        pos = np.array([
                            (i + bp[0]) * a,
                            (j + bp[1]) * b,
                            (k + bp[2]) * c
                        ])
                        atoms.append(pos)
        
        bonds = []
        for idx, pos in enumerate(atoms):
            for idx2, pos2 in enumerate(atoms[idx+1:], idx+1):
                dist = np.linalg.norm(pos - pos2)
                if 0.6 < dist < 0.9:
                    bonds.append((idx, idx2))
        
        return atoms, bonds


def render_crystal(atoms: List[np.ndarray],
                   bonds: List[Tuple[int, int]],
                   output_path: str,
                   title: str = "",
                   theme: str = 'crystal_blue',
                   atom_radius: float = 0.15,
                   bond_radius: float = 0.04,
                   view_angle: Tuple[float, float] = (25, 45),
                   figsize: Tuple[int, int] = (14, 14)):
    """
    Render a beautiful 3D crystal structure.
    """
    colors = THEMES.get(theme, THEMES['crystal_blue'])
    
    fig = plt.figure(figsize=figsize, facecolor=colors['bg'])
    ax = fig.add_subplot(111, projection='3d', facecolor=colors['bg'])
    
    # Get bounds
    all_coords = np.array(atoms)
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    z_range = all_coords[:, 2].max() - all_coords[:, 2].min()
    
    # Draw bonds first (behind atoms)
    bond_color = colors['bonds']
    for i, j in bonds:
        cyl_x, cyl_y, cyl_z = create_cylinder_points(
            atoms[i], atoms[j], bond_radius, resolution=8
        )
        if cyl_x is not None:
            ax.plot_surface(cyl_x, cyl_y, cyl_z, color=bond_color, 
                          alpha=0.8, shade=True, antialiased=True)
    
    # Draw atoms with gradient coloring based on position
    atom_colors = colors['atoms']
    for idx, pos in enumerate(atoms):
        # Color based on z-position for depth effect
        z_norm = (pos[2] - all_coords[:, 2].min()) / (z_range + 0.01)
        color_idx = int(z_norm * (len(atom_colors) - 1))
        color_idx = min(color_idx, len(atom_colors) - 1)
        
        sphere_x, sphere_y, sphere_z = create_sphere_points(
            pos, atom_radius, resolution=25
        )
        
        # Create shading effect
        base_color = mcolors.to_rgb(atom_colors[color_idx])
        glow_color = mcolors.to_rgb(colors['glow'])
        
        # Lighting based on position relative to view
        light_dir = np.array([1, 1, 1])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        ax.plot_surface(sphere_x, sphere_y, sphere_z,
                       color=atom_colors[color_idx],
                       alpha=0.95,
                       shade=True,
                       lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45),
                       antialiased=True)
    
    # Set axis properties
    ax.set_xlim(all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 0.5)
    ax.set_ylim(all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5)
    ax.set_zlim(all_coords[:, 2].min() - 0.5, all_coords[:, 2].max() + 0.5)
    
    # Equal aspect ratio
    max_range = max(x_range, y_range, z_range) / 2
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) / 2
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) / 2
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range - 0.3, mid_x + max_range + 0.3)
    ax.set_ylim(mid_y - max_range - 0.3, mid_y + max_range + 0.3)
    ax.set_zlim(mid_z - max_range - 0.3, mid_z + max_range + 0.3)
    
    # View angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Remove axes for clean look
    ax.set_axis_off()
    
    # Title
    if title:
        ax.set_title(title, color='white', fontsize=18, fontweight='bold',
                    pad=20, fontfamily='sans-serif')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight',
                facecolor=colors['bg'], edgecolor='none')
    plt.close()
    
    return True


def generate_all_crystal_structures(output_dir: Path, size: int = 3):
    """Generate beautiful renders for all crystal structure types."""
    
    print("\n" + "💎" * 30)
    print("  BEAUTIFUL CRYSTAL RENDERER")
    print("💎" * 30)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lattice = CrystalLattice(size=(size, size, size))
    
    structures = [
        ("cubic_P", lattice.generate_cubic_p, "Simple Cubic (Pm-3m)", "crystal_blue"),
        ("cubic_F_fcc", lattice.generate_cubic_f, "Face-Centered Cubic (Fm-3m)\nGold, Aluminum, Copper", "gold"),
        ("cubic_I_bcc", lattice.generate_cubic_i, "Body-Centered Cubic (Im-3m)\nIron, Tungsten", "ruby"),
        ("diamond", lattice.generate_diamond, "Diamond Cubic (Fd-3m)\nCarbon, Silicon", "diamond"),
        ("hexagonal_hcp", lattice.generate_hexagonal, "Hexagonal Close-Packed (P6₃/mmc)\nMagnesium, Zinc", "emerald"),
        ("tetragonal", lattice.generate_tetragonal, "Tetragonal (I4/mmm)", "sapphire"),
        ("trigonal", lattice.generate_trigonal, "Trigonal (R-3m)", "amethyst"),
        ("orthorhombic", lattice.generate_orthorhombic, "Orthorhombic (Pnma)", "sunset"),
    ]
    
    for name, generator, title, theme in structures:
        print(f"  Rendering {name}...", end=" ", flush=True)
        
        atoms, bonds = generator()
        output_path = output_dir / f"{name}.png"
        
        success = render_crystal(
            atoms, bonds,
            str(output_path),
            title=title,
            theme=theme,
            atom_radius=0.18,
            bond_radius=0.05,
            view_angle=(25, 35)
        )
        
        if success:
            print("✓")
        else:
            print("✗")
    
    print(f"\n✨ Crystal renders saved to: {output_dir}")


def generate_rotating_animation_frames(output_dir: Path, 
                                        structure_name: str = "diamond",
                                        num_frames: int = 36,
                                        size: int = 2):
    """Generate frames for a rotating crystal animation."""
    
    print(f"\n🎬 Generating {num_frames} rotation frames for {structure_name}...")
    
    frames_dir = output_dir / f"{structure_name}_rotation"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    lattice = CrystalLattice(size=(size, size, size))
    
    generators = {
        "cubic_P": lattice.generate_cubic_p,
        "cubic_F": lattice.generate_cubic_f,
        "cubic_I": lattice.generate_cubic_i,
        "diamond": lattice.generate_diamond,
        "hexagonal": lattice.generate_hexagonal,
    }
    
    if structure_name not in generators:
        print(f"Unknown structure: {structure_name}")
        return
    
    atoms, bonds = generators[structure_name]()
    
    for frame in range(num_frames):
        azim = frame * (360 / num_frames)
        output_path = frames_dir / f"frame_{frame:03d}.png"
        
        render_crystal(
            atoms, bonds,
            str(output_path),
            title="",
            theme='diamond',
            atom_radius=0.2,
            bond_radius=0.06,
            view_angle=(20, azim),
            figsize=(10, 10)
        )
        
        print(f"\r  Frame {frame + 1}/{num_frames}", end="", flush=True)
    
    print(f"\n✨ Frames saved to: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate beautiful crystal structure visualizations'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/crystals_beautiful',
        help='Output directory'
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=3,
        help='Lattice size (number of unit cells per dimension)'
    )
    parser.add_argument(
        '--animate',
        type=str,
        default=None,
        help='Generate rotation frames for specified structure'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=36,
        help='Number of animation frames'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.animate:
        generate_rotating_animation_frames(output_dir, args.animate, args.frames, args.size)
    else:
        generate_all_crystal_structures(output_dir, args.size)


if __name__ == "__main__":
    main()





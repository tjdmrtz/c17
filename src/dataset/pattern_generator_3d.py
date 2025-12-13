"""
3D Generator for the 17 Wallpaper Groups Extended to Three Dimensions

This module creates beautiful 3D volumetric patterns by extending the 17 wallpaper
groups (2D plane symmetry groups) into the third dimension using various methods:

- Layer stacking with phase shifts
- Helical extrusions
- Harmonic depth modulation
- Interference patterns

The resulting 3D patterns can be used for:
- Volumetric rendering
- 3D printing
- Texture synthesis
- Scientific visualization
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import gaussian_filter


class ExtrusionType(Enum):
    """Different methods to extend 2D patterns to 3D."""
    STACK = "stack"           # Simple layer stacking with phase shifts
    HELIX = "helix"           # Helical twist extrusion
    WAVE = "wave"             # Sinusoidal depth modulation
    INTERFERENCE = "interference"  # 3D wave interference patterns
    CRYSTAL = "crystal"       # Crystal-like 3D structure


@dataclass
class Wallpaper3DConfig:
    """Configuration for 3D wallpaper pattern generation."""
    resolution: Tuple[int, int, int]  # (x, y, z) dimensions
    extrusion_type: ExtrusionType
    twist_angle: float = 0.0          # For helix extrusion (radians per layer)
    phase_shift: float = 0.0          # Phase shift between layers
    depth_frequency: float = 1.0      # Frequency of depth modulation
    blend_layers: bool = True         # Smooth blending between layers


class WallpaperGroup3DGenerator:
    """
    Generates beautiful 3D volumetric patterns based on the 17 wallpaper groups.
    
    Each 2D wallpaper group is extended to 3D using sophisticated techniques
    that preserve and enhance the symmetry properties while creating visually
    stunning volumetric structures.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int, int] = (64, 64, 64),
                 seed: Optional[int] = None):
        """
        Initialize the 3D generator.
        
        Args:
            resolution: (x, y, z) size of the output volume
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        
        # Define the 17 wallpaper groups
        self.groups = [
            "p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm",
            "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"
        ]
        
    def _create_3d_motif(self, 
                         size: Tuple[int, int, int],
                         complexity: int = 4,
                         motif_type: str = "spherical") -> np.ndarray:
        """
        Create a random 3D motif (fundamental domain content).
        
        Args:
            size: (x, y, z) size of the motif
            complexity: Number of elements in the motif
            motif_type: Type of motif ("spherical", "crystalline", "organic", "mixed")
        
        Returns:
            3D array with the motif pattern
        """
        sx, sy, sz = size
        motif = np.zeros(size)
        
        if motif_type == "spherical":
            for _ in range(complexity):
                # Random 3D Gaussian blob
                cx = self.rng.random() * sx
                cy = self.rng.random() * sy
                cz = self.rng.random() * sz
                sigma = self.rng.random() * min(size) / 4 + min(size) / 8
                amplitude = self.rng.random() * 0.6 + 0.4
                
                z, y, x = np.ogrid[:sz, :sy, :sx]
                gaussian = amplitude * np.exp(
                    -((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / (2 * sigma**2)
                )
                motif += gaussian
                
        elif motif_type == "crystalline":
            for _ in range(complexity):
                # Crystal-like polyhedra approximated with intersecting planes
                cx = self.rng.random() * sx
                cy = self.rng.random() * sy
                cz = self.rng.random() * sz
                
                z, y, x = np.ogrid[:sz, :sy, :sx]
                
                # Create a crystal-like shape using multiple planes
                num_faces = self.rng.integers(4, 8)
                shape = np.ones(size)
                
                for _ in range(num_faces):
                    # Random plane normal
                    nx, ny, nz = self.rng.normal(size=3)
                    norm = np.sqrt(nx**2 + ny**2 + nz**2)
                    nx, ny, nz = nx/norm, ny/norm, nz/norm
                    
                    # Distance from center
                    dist = self.rng.random() * min(size) / 4 + 2
                    
                    # Plane equation
                    plane = nx * (x - cx) + ny * (y - cy) + nz * (z - cz) < dist
                    shape = shape * plane
                
                amplitude = self.rng.random() * 0.5 + 0.5
                motif += shape * amplitude
                
        elif motif_type == "organic":
            # Create organic-looking structures using noise
            for _ in range(complexity):
                cx = self.rng.random() * sx
                cy = self.rng.random() * sy
                cz = self.rng.random() * sz
                
                z, y, x = np.ogrid[:sz, :sy, :sx]
                
                # Base sphere
                radius = self.rng.random() * min(size) / 3 + min(size) / 6
                dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
                
                # Add some wobble using sin modulation
                freq = self.rng.random() * 2 + 0.5
                phase = self.rng.random() * 2 * np.pi
                theta = np.arctan2(y - cy, x - cx)
                phi = np.arctan2(np.sqrt((x - cx)**2 + (y - cy)**2), z - cz)
                wobble = 1 + 0.3 * np.sin(freq * theta + phase) * np.sin(freq * phi)
                
                blob = np.exp(-((dist / radius) * wobble)**2)
                motif += blob * (self.rng.random() * 0.5 + 0.5)
                
        elif motif_type == "mixed":
            # Combine different types
            motif = self._create_3d_motif(size, complexity // 2 + 1, "spherical")
            motif += self._create_3d_motif(size, complexity // 2 + 1, "crystalline")
            
        # Normalize
        if motif.max() > 0:
            motif = motif / motif.max()
            
        return motif
    
    def _rotate_3d(self, volume: np.ndarray, 
                   angle: float, 
                   axes: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """Rotate 3D volume around specified axes."""
        angle_deg = np.degrees(angle)
        return scipy_rotate(volume, angle_deg, axes=axes, reshape=False, order=1, mode='wrap')
    
    def _reflect_3d(self, volume: np.ndarray, axis: int) -> np.ndarray:
        """Reflect 3D volume across specified axis."""
        return np.flip(volume, axis=axis)
    
    def _glide_3d(self, volume: np.ndarray, axis: int, 
                  reflect_axis: int, shift: float = 0.5) -> np.ndarray:
        """3D glide reflection: reflect and translate."""
        reflected = self._reflect_3d(volume, reflect_axis)
        shift_pixels = int(shift * volume.shape[axis])
        return np.roll(reflected, shift_pixels, axis=axis)
    
    def _tile_3d(self, cell: np.ndarray, 
                 tiles: Tuple[int, int, int]) -> np.ndarray:
        """Tile the 3D unit cell."""
        return np.tile(cell, tiles)
    
    def _apply_z_modulation(self, volume: np.ndarray, 
                            modulation_type: str = "wave",
                            frequency: float = 2.0) -> np.ndarray:
        """Apply depth-based modulation for visual interest."""
        sz, sy, sx = volume.shape
        z = np.arange(sz).reshape(-1, 1, 1)
        
        if modulation_type == "wave":
            # Sinusoidal modulation
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * z / sz)
        elif modulation_type == "gradient":
            # Linear gradient
            mod = z / sz
        elif modulation_type == "pulse":
            # Gaussian pulses
            mod = np.exp(-((z - sz/2)**2) / (sz/4)**2)
        else:
            mod = 1.0
            
        return volume * mod
    
    def _apply_twist(self, volume: np.ndarray, 
                     twist_per_layer: float = 0.05) -> np.ndarray:
        """Apply helical twist to the volume."""
        sz, sy, sx = volume.shape
        result = np.zeros_like(volume)
        
        for z in range(sz):
            angle = twist_per_layer * z
            layer = volume[z]
            rotated = scipy_rotate(layer, np.degrees(angle), reshape=False, order=1, mode='wrap')
            result[z] = rotated
            
        return result
    
    def _create_interference_pattern(self, 
                                     frequencies: List[Tuple[float, float, float]],
                                     phases: List[float] = None) -> np.ndarray:
        """Create 3D interference pattern from multiple waves."""
        sx, sy, sz = self.resolution
        
        if phases is None:
            phases = [0.0] * len(frequencies)
        
        z, y, x = np.ogrid[:sz, :sy, :sx]
        pattern = np.zeros((sz, sy, sx))
        
        for (fx, fy, fz), phase in zip(frequencies, phases):
            wave = np.sin(2*np.pi*(fx*x/sx + fy*y/sy + fz*z/sz) + phase)
            pattern += wave
            
        # Normalize to [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
        return pattern
    
    # === 3D Pattern Generators for Each Wallpaper Group ===
    
    def generate_p1_3d(self, motif_size: int = 24, 
                       extrusion: ExtrusionType = ExtrusionType.WAVE,
                       **kwargs) -> np.ndarray:
        """p1 in 3D: Pure translation symmetry in all directions."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        # Create 3D motif
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Tile in all directions
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(motif, tiles)
        
        # Apply modulation for visual interest
        pattern = self._apply_z_modulation(pattern[:sz, :sy, :sx], "wave", 2.0)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p2_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.HELIX,
                       **kwargs) -> np.ndarray:
        """p2 in 3D: 180° rotation centers extended with helical twist."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Create unit cell with 180° rotation around z-axis
        rotated = self._rotate_3d(motif, np.pi, axes=(1, 2))
        
        # Combine in a 2x2 arrangement
        top = np.concatenate([motif, rotated], axis=2)
        bottom = np.concatenate([rotated, motif], axis=2)
        cell = np.concatenate([top, bottom], axis=1)
        
        # Tile
        tiles = (sz // cell.shape[0] + 1, 
                 sy // cell.shape[1] + 1, 
                 sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        # Apply helical twist for visual interest
        if extrusion == ExtrusionType.HELIX:
            pattern = self._apply_twist(pattern[:sz, :sy, :sx], 0.02)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_pm_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.WAVE,
                       **kwargs) -> np.ndarray:
        """pm in 3D: Parallel reflection planes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        # Remove extrusion from kwargs before passing to _create_3d_motif
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Reflect across x-axis (yz plane)
        reflected = self._reflect_3d(motif, axis=2)
        cell = np.concatenate([motif, reflected], axis=2)
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        # Add wave modulation
        pattern = self._apply_z_modulation(pattern[:sz, :sy, :sx], "wave", 3.0)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_pg_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.WAVE,
                       **kwargs) -> np.ndarray:
        """pg in 3D: Parallel glide planes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Glide reflection
        glided = self._glide_3d(motif, axis=0, reflect_axis=1)
        cell = np.concatenate([motif, glided], axis=2)
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_cm_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.WAVE,
                       **kwargs) -> np.ndarray:
        """cm in 3D: Reflection planes with glide between layers."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Reflect
        reflected = self._reflect_3d(motif, axis=2)
        
        # First layer
        row1 = np.concatenate([motif, reflected], axis=2)
        # Second layer (shifted)
        row2 = np.roll(row1, ms, axis=2)
        cell = np.concatenate([row1, row2], axis=1)
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_pmm_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """pmm in 3D: Perpendicular reflection planes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Create 2x2x2 cell with all reflections
        ref_x = self._reflect_3d(motif, axis=2)
        ref_y = self._reflect_3d(motif, axis=1)
        ref_xy = self._reflect_3d(ref_x, axis=1)
        
        # XY plane
        top = np.concatenate([motif, ref_x], axis=2)
        bottom = np.concatenate([ref_y, ref_xy], axis=2)
        layer = np.concatenate([top, bottom], axis=1)
        
        # Z reflection
        ref_z = self._reflect_3d(layer, axis=0)
        cell = np.concatenate([layer, ref_z], axis=0)
        
        # Tile
        tiles = (sz // cell.shape[0] + 1, 
                 sy // cell.shape[1] + 1, 
                 sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_pmg_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """pmg in 3D: Reflection + perpendicular glide planes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Reflection along x
        reflected = self._reflect_3d(motif, axis=2)
        top = np.concatenate([motif, reflected], axis=2)
        
        # Glide reflection for bottom
        glided_motif = self._glide_3d(motif, axis=1, reflect_axis=2)
        glided_reflected = self._glide_3d(reflected, axis=1, reflect_axis=2)
        bottom = np.concatenate([glided_motif, glided_reflected], axis=2)
        
        cell = np.concatenate([top, bottom], axis=1)
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_pgg_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """pgg in 3D: Perpendicular glide planes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Two perpendicular glide reflections
        glided_x = self._glide_3d(motif, axis=2, reflect_axis=1)
        top = np.concatenate([motif, glided_x], axis=2)
        
        glided_y = self._glide_3d(motif, axis=1, reflect_axis=2)
        glided_xy = self._glide_3d(glided_y, axis=2, reflect_axis=1)
        bottom = np.concatenate([glided_y, glided_xy], axis=2)
        
        cell = np.concatenate([top, bottom], axis=1)
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_cmm_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """cmm in 3D: Centered cell with reflection planes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Create rhombic-like cell with 2-fold rotation and reflections
        ref_x = self._reflect_3d(motif, axis=2)
        ref_y = self._reflect_3d(motif, axis=1)
        ref_xy = self._reflect_3d(ref_x, axis=1)
        
        # 2x2 cell
        top = np.concatenate([motif, ref_x], axis=2)
        bottom = np.concatenate([ref_y, ref_xy], axis=2)
        basic_cell = np.concatenate([top, bottom], axis=1)
        
        # Shift for centering
        shifted = np.roll(np.roll(basic_cell, ms, axis=1), ms, axis=2)
        cell = (basic_cell + shifted * 0.5)
        if cell.max() > 0:
            cell = cell / cell.max()
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p4_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.HELIX,
                       **kwargs) -> np.ndarray:
        """p4 in 3D: 90° rotation axes (beautiful helical structures)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Create 2x2 cell with 90° rotations around z-axis
        rot90 = self._rotate_3d(motif, np.pi/2, axes=(1, 2))
        rot180 = self._rotate_3d(motif, np.pi, axes=(1, 2))
        rot270 = self._rotate_3d(motif, 3*np.pi/2, axes=(1, 2))
        
        top = np.concatenate([motif, rot90], axis=2)
        bottom = np.concatenate([rot270, rot180], axis=2)
        cell = np.concatenate([top, bottom], axis=1)
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        # Apply helical twist for stunning effect
        if extrusion == ExtrusionType.HELIX:
            pattern = self._apply_twist(pattern[:sz, :sy, :sx], np.pi/32)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p4m_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """p4m in 3D: Square with reflection planes on all axes."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        # Use smaller motif for fundamental domain
        fund_size = ms // 2
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((fund_size, fund_size, fund_size), **kwargs)
        
        # Apply diagonal symmetry
        for i in range(fund_size):
            for j in range(fund_size):
                for k in range(fund_size):
                    if j > i:
                        motif[k, i, j] = motif[k, i, i]
        
        # Build up symmetry
        reflected = self._reflect_3d(motif, axis=2)
        quarter = np.concatenate([motif, reflected], axis=2)
        
        reflected_v = self._reflect_3d(quarter, axis=1)
        half = np.concatenate([quarter, reflected_v], axis=1)
        
        # 4-fold rotation
        rot90 = self._rotate_3d(half, np.pi/2, axes=(1, 2))
        rot180 = self._rotate_3d(half, np.pi, axes=(1, 2))
        rot270 = self._rotate_3d(half, 3*np.pi/2, axes=(1, 2))
        
        # Match sizes
        size = min(half.shape[1], rot90.shape[1])
        half = half[:, :size, :size]
        rot90 = rot90[:, :size, :size]
        rot180 = rot180[:, :size, :size]
        rot270 = rot270[:, :size, :size]
        
        top = np.concatenate([half, rot90], axis=2)
        bottom = np.concatenate([rot270, rot180], axis=2)
        cell = np.concatenate([top, bottom], axis=1)
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p4g_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """p4g in 3D: Square with glide planes and rotations."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # 90° rotations with glide reflections
        rot90 = self._rotate_3d(motif, np.pi/2, axes=(1, 2))
        rot180 = self._rotate_3d(motif, np.pi, axes=(1, 2))
        rot270 = self._rotate_3d(motif, 3*np.pi/2, axes=(1, 2))
        
        # Apply glide to alternating cells
        top = np.concatenate([motif, self._reflect_3d(rot90, axis=2)], axis=2)
        bottom = np.concatenate([self._reflect_3d(rot270, axis=1), rot180], axis=2)
        cell = np.concatenate([top, bottom], axis=1)
        
        # Tile
        tiles = (sz // ms + 1, sy // cell.shape[1] + 1, sx // cell.shape[2] + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p3_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.HELIX,
                       **kwargs) -> np.ndarray:
        """p3 in 3D: 120° rotation axes (trigonal symmetry)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # 3-fold rotation around z
        rot120 = self._rotate_3d(motif, 2*np.pi/3, axes=(1, 2))
        rot240 = self._rotate_3d(motif, 4*np.pi/3, axes=(1, 2))
        
        # Combine
        cell = motif + rot120 + rot240
        if cell.max() > 0:
            cell = cell / cell.max()
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        # Apply helical twist for beautiful effect
        if extrusion == ExtrusionType.HELIX:
            pattern = self._apply_twist(pattern[:sz, :sy, :sx], np.pi/48)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p3m1_3d(self, motif_size: int = 24,
                         extrusion: ExtrusionType = ExtrusionType.WAVE,
                         **kwargs) -> np.ndarray:
        """p3m1 in 3D: 120° rotation with reflection planes through centers."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Apply reflection before rotation
        reflected = self._reflect_3d(motif, axis=2)
        combined = (motif + reflected) / 2
        
        rot120 = self._rotate_3d(combined, 2*np.pi/3, axes=(1, 2))
        rot240 = self._rotate_3d(combined, 4*np.pi/3, axes=(1, 2))
        
        cell = combined + rot120 + rot240
        if cell.max() > 0:
            cell = cell / cell.max()
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p31m_3d(self, motif_size: int = 24,
                         extrusion: ExtrusionType = ExtrusionType.WAVE,
                         **kwargs) -> np.ndarray:
        """p31m in 3D: 120° rotation with reflection planes between centers."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # 3-fold rotation first
        rot120 = self._rotate_3d(motif, 2*np.pi/3, axes=(1, 2))
        rot240 = self._rotate_3d(motif, 4*np.pi/3, axes=(1, 2))
        
        combined = motif + rot120 + rot240
        
        # Then reflection
        reflected = self._reflect_3d(combined, axis=1)
        cell = (combined + reflected) / 2
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p6_3d(self, motif_size: int = 24,
                       extrusion: ExtrusionType = ExtrusionType.HELIX,
                       **kwargs) -> np.ndarray:
        """p6 in 3D: 60° rotation axes (hexagonal symmetry)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # 6-fold rotation
        cell = motif.copy()
        for k in range(1, 6):
            rotated = self._rotate_3d(motif, k * np.pi / 3, axes=(1, 2))
            cell = cell + rotated
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        # Apply helical twist
        if extrusion == ExtrusionType.HELIX:
            pattern = self._apply_twist(pattern[:sz, :sy, :sx], np.pi/36)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_p6m_3d(self, motif_size: int = 24,
                        extrusion: ExtrusionType = ExtrusionType.WAVE,
                        **kwargs) -> np.ndarray:
        """p6m in 3D: Hexagonal with all symmetries (most symmetric)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        kwargs.pop('extrusion', None)
        motif = self._create_3d_motif((ms, ms, ms), **kwargs)
        
        # Apply reflection first
        reflected = self._reflect_3d(motif, axis=2)
        base = (motif + reflected) / 2
        
        # 6-fold rotation
        cell = base.copy()
        for k in range(1, 6):
            rotated = self._rotate_3d(base, k * np.pi / 3, axes=(1, 2))
            cell = cell + rotated
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        # Tile
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate(self, 
                 group_name: str, 
                 motif_size: int = 24,
                 extrusion: ExtrusionType = ExtrusionType.WAVE,
                 **kwargs) -> np.ndarray:
        """
        Generate a 3D pattern for a specific wallpaper group.
        
        Args:
            group_name: Name of the wallpaper group (p1, p2, pm, etc.)
            motif_size: Size of the fundamental 3D motif
            extrusion: Type of 3D extrusion to apply
            **kwargs: Additional arguments for motif creation
            
        Returns:
            3D numpy array with the generated volumetric pattern
        """
        generators = {
            "p1": self.generate_p1_3d,
            "p2": self.generate_p2_3d,
            "pm": self.generate_pm_3d,
            "pg": self.generate_pg_3d,
            "cm": self.generate_cm_3d,
            "pmm": self.generate_pmm_3d,
            "pmg": self.generate_pmg_3d,
            "pgg": self.generate_pgg_3d,
            "cmm": self.generate_cmm_3d,
            "p4": self.generate_p4_3d,
            "p4m": self.generate_p4m_3d,
            "p4g": self.generate_p4g_3d,
            "p3": self.generate_p3_3d,
            "p3m1": self.generate_p3m1_3d,
            "p31m": self.generate_p31m_3d,
            "p6": self.generate_p6_3d,
            "p6m": self.generate_p6m_3d,
        }
        
        if group_name not in generators:
            raise ValueError(f"Unknown wallpaper group: {group_name}. "
                           f"Valid groups: {list(generators.keys())}")
        
        return generators[group_name](motif_size, extrusion=extrusion, **kwargs)
    
    def generate_all(self, 
                     motif_size: int = 24,
                     extrusion: ExtrusionType = ExtrusionType.WAVE,
                     **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate 3D patterns for all 17 wallpaper groups.
        
        Returns:
            Dictionary mapping group names to 3D pattern arrays
        """
        return {name: self.generate(name, motif_size, extrusion, **kwargs) 
                for name in self.groups}
    
    def generate_with_interference(self, 
                                   group_name: str,
                                   num_waves: int = 6) -> np.ndarray:
        """
        Generate pattern using 3D wave interference based on group symmetry.
        
        This creates particularly beautiful, gem-like patterns.
        """
        sx, sy, sz = self.resolution
        
        # Define wave directions based on group symmetry
        if group_name in ["p4", "p4m", "p4g"]:
            # 4-fold symmetry
            angles = [k * np.pi/2 for k in range(4)]
        elif group_name in ["p3", "p3m1", "p31m"]:
            # 3-fold symmetry
            angles = [k * 2*np.pi/3 for k in range(3)]
        elif group_name in ["p6", "p6m"]:
            # 6-fold symmetry
            angles = [k * np.pi/3 for k in range(6)]
        else:
            # 2-fold or lower
            angles = [0, np.pi/2]
        
        # Generate wave frequencies
        frequencies = []
        phases = []
        
        for angle in angles:
            fx = np.cos(angle)
            fy = np.sin(angle)
            fz = self.rng.random() * 0.5  # Some z variation
            frequencies.append((fx * 2, fy * 2, fz))
            phases.append(self.rng.random() * 2 * np.pi)
        
        pattern = self._create_interference_pattern(frequencies, phases)
        
        # Apply group-specific post-processing
        if "m" in group_name:
            # Add reflection symmetry
            pattern = (pattern + np.flip(pattern, axis=2)) / 2
        
        return pattern
    
    @staticmethod
    def list_groups() -> List[str]:
        """List all available wallpaper groups."""
        return ["p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm",
                "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"]


def visualize_3d_pattern(pattern: np.ndarray, 
                         output_path: str,
                         threshold: float = 0.3,
                         colormap: str = "viridis"):
    """
    Visualize a 3D pattern using matplotlib's 3D rendering.
    
    Args:
        pattern: 3D numpy array
        output_path: Path to save the visualization
        threshold: Isosurface threshold for rendering
        colormap: Matplotlib colormap to use
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create voxel visualization
    voxels = pattern > threshold
    
    # Color based on z-coordinate for depth
    colors = np.zeros(voxels.shape + (4,))
    cmap = plt.cm.get_cmap(colormap)
    
    for i in range(pattern.shape[0]):
        color = cmap(i / pattern.shape[0])
        colors[i, :, :] = color
    
    ax.voxels(voxels, facecolors=colors, edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_slices_visualization(pattern: np.ndarray,
                                output_path: str,
                                num_slices: int = 9):
    """
    Create a grid of 2D slices through the 3D pattern.
    
    Args:
        pattern: 3D numpy array
        output_path: Path to save the visualization
        num_slices: Number of slices to show
    """
    import matplotlib.pyplot as plt
    
    # Calculate grid size
    n = int(np.ceil(np.sqrt(num_slices)))
    
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    axes = axes.flatten()
    
    # Get evenly spaced slice indices
    z_indices = np.linspace(0, pattern.shape[0] - 1, num_slices, dtype=int)
    
    for idx, (ax, z) in enumerate(zip(axes, z_indices)):
        slice_data = pattern[z]
        im = ax.imshow(slice_data, cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'Z = {z}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('3D Pattern - Z Slices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Demo: Generate all 17 groups in 3D
    print("3D Wallpaper Group Generator Demo")
    print("=" * 50)
    
    generator = WallpaperGroup3DGenerator(resolution=(48, 48, 48), seed=42)
    
    for group_name in generator.list_groups():
        print(f"Generating 3D pattern for {group_name}...")
        pattern = generator.generate(group_name, motif_size=16, complexity=3)
        print(f"  Shape: {pattern.shape}, Range: [{pattern.min():.3f}, {pattern.max():.3f}]")
    
    print("\nDone!")


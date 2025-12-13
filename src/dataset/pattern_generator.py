"""
Generator for the 17 Wallpaper Groups (Plane Crystallographic Groups)

The 17 wallpaper groups are the only distinct ways to tile a 2D plane with
a repeating pattern. Each group is defined by its symmetry operations:
- Translations
- Rotations (2-fold, 3-fold, 4-fold, 6-fold)
- Reflections
- Glide reflections

Groups by lattice type:
- Oblique: p1, p2
- Rectangular: pm, pg, pmm, pmg, pgg, cm, cmm
- Square: p4, p4m, p4g
- Hexagonal: p3, p3m1, p31m, p6, p6m
"""

import numpy as np
from typing import Tuple, List, Callable, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class LatticeType(Enum):
    OBLIQUE = "oblique"
    RECTANGULAR = "rectangular"
    SQUARE = "square"
    HEXAGONAL = "hexagonal"


@dataclass
class WallpaperGroup:
    """Represents a wallpaper group with its properties."""
    name: str
    lattice_type: LatticeType
    rotation_order: int  # Maximum rotation symmetry (1, 2, 3, 4, or 6)
    has_reflection: bool
    has_glide: bool
    description: str


# Definition of all 17 wallpaper groups
WALLPAPER_GROUPS = {
    # Oblique lattice
    "p1": WallpaperGroup("p1", LatticeType.OBLIQUE, 1, False, False,
                         "Only translations, no point symmetry"),
    "p2": WallpaperGroup("p2", LatticeType.OBLIQUE, 2, False, False,
                         "180° rotation centers"),
    
    # Rectangular lattice
    "pm": WallpaperGroup("pm", LatticeType.RECTANGULAR, 1, True, False,
                         "Parallel reflection axes"),
    "pg": WallpaperGroup("pg", LatticeType.RECTANGULAR, 1, False, True,
                         "Parallel glide reflections"),
    "cm": WallpaperGroup("cm", LatticeType.RECTANGULAR, 1, True, True,
                         "Reflection axes with glide between"),
    "pmm": WallpaperGroup("pmm", LatticeType.RECTANGULAR, 2, True, False,
                          "Perpendicular reflection axes"),
    "pmg": WallpaperGroup("pmg", LatticeType.RECTANGULAR, 2, True, True,
                          "Reflection + perpendicular glide"),
    "pgg": WallpaperGroup("pgg", LatticeType.RECTANGULAR, 2, False, True,
                          "Perpendicular glide reflections"),
    "cmm": WallpaperGroup("cmm", LatticeType.RECTANGULAR, 2, True, True,
                          "Centered cell with reflections"),
    
    # Square lattice
    "p4": WallpaperGroup("p4", LatticeType.SQUARE, 4, False, False,
                         "90° rotation centers"),
    "p4m": WallpaperGroup("p4m", LatticeType.SQUARE, 4, True, True,
                          "Square with reflections on all axes"),
    "p4g": WallpaperGroup("p4g", LatticeType.SQUARE, 4, True, True,
                          "Square with glides and rotations"),
    
    # Hexagonal lattice
    "p3": WallpaperGroup("p3", LatticeType.HEXAGONAL, 3, False, False,
                         "120° rotation centers"),
    "p3m1": WallpaperGroup("p3m1", LatticeType.HEXAGONAL, 3, True, False,
                           "120° rotation with reflection axes through centers"),
    "p31m": WallpaperGroup("p31m", LatticeType.HEXAGONAL, 3, True, False,
                           "120° rotation with reflection axes between centers"),
    "p6": WallpaperGroup("p6", LatticeType.HEXAGONAL, 6, False, False,
                         "60° rotation centers"),
    "p6m": WallpaperGroup("p6m", LatticeType.HEXAGONAL, 6, True, True,
                          "Hexagonal with all symmetries"),
}


class WallpaperGroupGenerator:
    """
    Generates patterns for all 17 wallpaper groups.
    
    The generator creates a fundamental domain (asymmetric unit) and then
    applies the symmetry operations of each group to tile the plane.
    """
    
    def __init__(self, resolution: int = 256, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            resolution: Size of the output image (resolution x resolution)
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        
    def _create_motif(self, 
                      size: int, 
                      complexity: int = 3,
                      motif_type: str = "gaussian") -> np.ndarray:
        """
        Create a random motif (fundamental domain content).
        
        Args:
            size: Size of the motif
            complexity: Number of elements in the motif
            motif_type: Type of motif ("gaussian", "geometric", "mixed")
        
        Returns:
            2D array with the motif pattern
        """
        motif = np.zeros((size, size))
        
        if motif_type == "gaussian":
            for _ in range(complexity):
                # Random Gaussian blob
                cx, cy = self.rng.random(2) * size
                sigma = self.rng.random() * size / 4 + size / 10
                amplitude = self.rng.random() * 0.5 + 0.5
                
                y, x = np.ogrid[:size, :size]
                gaussian = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                motif += gaussian
                
        elif motif_type == "geometric":
            for _ in range(complexity):
                shape = self.rng.choice(["circle", "line", "triangle"])
                if shape == "circle":
                    cx, cy = self.rng.random(2) * size
                    radius = self.rng.random() * size / 4 + 2
                    y, x = np.ogrid[:size, :size]
                    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                    motif[mask] += self.rng.random() * 0.7 + 0.3
                elif shape == "line":
                    angle = self.rng.random() * np.pi
                    thickness = int(self.rng.random() * 3 + 1)
                    y, x = np.ogrid[:size, :size]
                    cx, cy = size / 2, size / 2
                    dist = np.abs(np.cos(angle) * (x - cx) + np.sin(angle) * (y - cy))
                    motif[dist < thickness] += self.rng.random() * 0.7 + 0.3
                elif shape == "triangle":
                    # Simple triangle using barycentric approach
                    p1 = self.rng.random(2) * size
                    p2 = self.rng.random(2) * size
                    p3 = self.rng.random(2) * size
                    y, x = np.ogrid[:size, :size]
                    # Simplified: just use a polygon mask approximation
                    cx, cy = (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3
                    radius = size / 6
                    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                    motif[mask] += self.rng.random() * 0.7 + 0.3
                    
        elif motif_type == "mixed":
            motif = self._create_motif(size, complexity // 2 + 1, "gaussian")
            motif += self._create_motif(size, complexity // 2 + 1, "geometric")
            
        # Normalize
        if motif.max() > 0:
            motif = motif / motif.max()
            
        return motif
    
    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (in radians) around center."""
        from scipy.ndimage import rotate as scipy_rotate
        angle_deg = np.degrees(angle)
        return scipy_rotate(image, angle_deg, reshape=False, order=1, mode='constant')
    
    def _reflect_x(self, image: np.ndarray) -> np.ndarray:
        """Reflect image across vertical axis (x-reflection)."""
        return np.flip(image, axis=1)
    
    def _reflect_y(self, image: np.ndarray) -> np.ndarray:
        """Reflect image across horizontal axis (y-reflection)."""
        return np.flip(image, axis=0)
    
    def _glide_x(self, image: np.ndarray, shift: float = 0.5) -> np.ndarray:
        """Glide reflection: reflect and translate along x."""
        reflected = self._reflect_y(image)
        shift_pixels = int(shift * image.shape[1])
        return np.roll(reflected, shift_pixels, axis=1)
    
    def _glide_y(self, image: np.ndarray, shift: float = 0.5) -> np.ndarray:
        """Glide reflection: reflect and translate along y."""
        reflected = self._reflect_x(image)
        shift_pixels = int(shift * image.shape[0])
        return np.roll(reflected, shift_pixels, axis=0)
    
    def _tile_pattern(self, 
                      cell: np.ndarray, 
                      tiles_x: int = 4, 
                      tiles_y: int = 4) -> np.ndarray:
        """Tile the unit cell to create the full pattern."""
        return np.tile(cell, (tiles_y, tiles_x))
    
    def _apply_hexagonal_lattice(self, cell: np.ndarray, tiles: int = 4) -> np.ndarray:
        """Apply hexagonal lattice tiling."""
        h, w = cell.shape
        # Create a larger canvas
        out_h = int(h * tiles * np.sqrt(3) / 2)
        out_w = w * tiles
        output = np.zeros((out_h, out_w))
        
        for row in range(tiles * 2):
            for col in range(tiles):
                y_offset = int(row * h * np.sqrt(3) / 4)
                x_offset = col * w + (row % 2) * (w // 2)
                
                if y_offset + h <= out_h and x_offset + w <= out_w:
                    output[y_offset:y_offset+h, x_offset:x_offset+w] += cell
                    
        return output[:self.resolution, :self.resolution]
    
    # === Pattern generators for each wallpaper group ===
    
    def generate_p1(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p1: Only translations (no point symmetry)."""
        motif = self._create_motif(motif_size, **kwargs)
        tiles = self.resolution // motif_size + 1
        pattern = self._tile_pattern(motif, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p2(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p2: 180° rotation centers."""
        motif = self._create_motif(motif_size, **kwargs)
        # Create unit cell with 180° rotation
        rotated = self._rotate(motif, np.pi)
        # Combine original and rotated in a 2x2 arrangement
        top = np.hstack([motif, rotated])
        bottom = np.hstack([rotated, motif])
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_pm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """pm: Parallel reflection axes."""
        motif = self._create_motif(motif_size, **kwargs)
        reflected = self._reflect_x(motif)
        cell = np.hstack([motif, reflected])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_pg(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """pg: Parallel glide reflections."""
        motif = self._create_motif(motif_size, **kwargs)
        glided = self._glide_x(motif)
        cell = np.hstack([motif, glided])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_cm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """cm: Reflection axes with glide between."""
        motif = self._create_motif(motif_size, **kwargs)
        reflected = self._reflect_x(motif)
        
        # First row: motif | reflected
        row1 = np.hstack([motif, reflected])
        # Second row: shifted version
        row2 = np.roll(row1, motif_size, axis=1)
        cell = np.vstack([row1, row2])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_pmm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """pmm: Perpendicular reflection axes."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Create 2x2 cell with all reflections
        top = np.hstack([motif, self._reflect_x(motif)])
        bottom = np.hstack([self._reflect_y(motif), 
                          self._reflect_x(self._reflect_y(motif))])
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_pmg(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """pmg: Reflection + perpendicular glide."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Reflection along x
        reflected = self._reflect_x(motif)
        top = np.hstack([motif, reflected])
        
        # Glide reflection for bottom
        glided_motif = self._glide_y(motif)
        glided_reflected = self._glide_y(reflected)
        bottom = np.hstack([glided_motif, glided_reflected])
        
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_pgg(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """pgg: Perpendicular glide reflections."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Two perpendicular glide reflections
        glided_x = self._glide_x(motif)
        top = np.hstack([motif, glided_x])
        
        glided_y = self._glide_y(motif)
        glided_xy = self._glide_x(glided_y)
        bottom = np.hstack([glided_y, glided_xy])
        
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_cmm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """cmm: Centered cell with reflections."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Create rhombic-like cell with 2-fold rotation and reflections
        reflected_x = self._reflect_x(motif)
        reflected_y = self._reflect_y(motif)
        reflected_xy = self._reflect_x(reflected_y)
        
        # 2x2 cell
        top = np.hstack([motif, reflected_x])
        bottom = np.hstack([reflected_y, reflected_xy])
        basic_cell = np.vstack([top, bottom])
        
        # Shift for centering
        h, w = basic_cell.shape
        shifted = np.roll(np.roll(basic_cell, h//2, axis=0), w//2, axis=1)
        cell = (basic_cell + shifted * 0.5)
        cell = cell / cell.max() if cell.max() > 0 else cell
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p4(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p4: 90° rotation centers."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Create 2x2 cell with 90° rotations
        rot90 = self._rotate(motif, np.pi/2)
        rot180 = self._rotate(motif, np.pi)
        rot270 = self._rotate(motif, 3*np.pi/2)
        
        top = np.hstack([motif, rot90])
        bottom = np.hstack([rot270, rot180])
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p4m(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p4m: Square with reflections on all axes."""
        # Use smaller motif for the fundamental domain (1/8 of the cell)
        fund_size = motif_size // 2
        motif = self._create_motif(fund_size, **kwargs)
        
        # Create triangular fundamental domain and reflect
        # Simplified: use quarter then apply 4-fold + reflections
        h, w = motif.shape
        
        # Make upper triangle
        for i in range(h):
            for j in range(w):
                if j > i:
                    motif[i, j] = motif[i, i]
        
        reflected = self._reflect_x(motif)
        quarter = np.hstack([motif, reflected[:, 1:]])
        
        # Reflect to make half
        quarter_h = quarter.shape[0]
        reflected_v = self._reflect_y(quarter)
        half = np.vstack([quarter, reflected_v[1:, :]])
        
        # Apply 4-fold rotation
        rot90 = self._rotate(half, np.pi/2)
        rot180 = self._rotate(half, np.pi)
        rot270 = self._rotate(half, 3*np.pi/2)
        
        # Resize to match
        size = min(half.shape[0], rot90.shape[0])
        half = half[:size, :size]
        rot90 = rot90[:size, :size]
        rot180 = rot180[:size, :size]
        rot270 = rot270[:size, :size]
        
        top = np.hstack([half, rot90])
        bottom = np.hstack([rot270, rot180])
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p4g(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p4g: Square with glides and rotations."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # 90° rotations with glide reflections
        rot90 = self._rotate(motif, np.pi/2)
        rot180 = self._rotate(motif, np.pi)
        rot270 = self._rotate(motif, 3*np.pi/2)
        
        # Apply glide to alternating cells
        top = np.hstack([motif, self._reflect_x(rot90)])
        bottom = np.hstack([self._reflect_y(rot270), rot180])
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p3(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p3: 120° rotation centers (hexagonal)."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Create hexagonal cell with 3-fold rotation
        rot120 = self._rotate(motif, 2*np.pi/3)
        rot240 = self._rotate(motif, 4*np.pi/3)
        
        # Combine with offset for hexagonal packing
        h, w = motif.shape
        cell = np.zeros((int(h * 1.5), int(w * 1.5)))
        
        # Place three rotated copies
        cell[:h, :w] += motif
        offset = h // 2
        cell[offset:offset+h, w//3:w//3+w] += rot120[:h, :w]
        cell[:h, w//2:w//2+w] += rot240[:h, :w]
        
        cell = cell[:h, :w]
        if cell.max() > 0:
            cell = cell / cell.max()
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p3m1(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p3m1: 120° rotation with reflection axes through rotation centers."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Apply reflection before rotation
        reflected = self._reflect_x(motif)
        combined = (motif + reflected) / 2
        
        rot120 = self._rotate(combined, 2*np.pi/3)
        rot240 = self._rotate(combined, 4*np.pi/3)
        
        h, w = combined.shape
        cell = combined.copy()
        cell += rot120[:h, :w]
        cell += rot240[:h, :w]
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p31m(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p31m: 120° rotation with reflection axes between rotation centers."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # 3-fold rotation first, then reflection
        rot120 = self._rotate(motif, 2*np.pi/3)
        rot240 = self._rotate(motif, 4*np.pi/3)
        
        h, w = motif.shape
        combined = motif + rot120[:h, :w] + rot240[:h, :w]
        
        # Apply reflection with different axis orientation
        reflected = self._reflect_y(combined)
        cell = (combined + reflected) / 2
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p6(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p6: 60° rotation centers (hexagonal)."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # 6-fold rotation
        cell = motif.copy()
        for k in range(1, 6):
            rotated = self._rotate(motif, k * np.pi / 3)
            h, w = min(cell.shape[0], rotated.shape[0]), min(cell.shape[1], rotated.shape[1])
            cell[:h, :w] += rotated[:h, :w]
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate_p6m(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """p6m: Hexagonal with all symmetries."""
        motif = self._create_motif(motif_size, **kwargs)
        
        # Apply reflection first
        reflected = self._reflect_x(motif)
        base = (motif + reflected) / 2
        
        # 6-fold rotation
        cell = base.copy()
        for k in range(1, 6):
            rotated = self._rotate(base, k * np.pi / 3)
            h, w = min(cell.shape[0], rotated.shape[0]), min(cell.shape[1], rotated.shape[1])
            cell[:h, :w] += rotated[:h, :w]
        
        if cell.max() > 0:
            cell = cell / cell.max()
        
        tiles = self.resolution // cell.shape[0] + 1
        pattern = self._tile_pattern(cell, tiles, tiles)
        return pattern[:self.resolution, :self.resolution]
    
    def generate(self, 
                 group_name: str, 
                 motif_size: int = 64, 
                 **kwargs) -> np.ndarray:
        """
        Generate a pattern for a specific wallpaper group.
        
        Args:
            group_name: Name of the wallpaper group (p1, p2, pm, etc.)
            motif_size: Size of the fundamental motif
            **kwargs: Additional arguments for motif creation
            
        Returns:
            2D numpy array with the generated pattern
        """
        generators = {
            "p1": self.generate_p1,
            "p2": self.generate_p2,
            "pm": self.generate_pm,
            "pg": self.generate_pg,
            "cm": self.generate_cm,
            "pmm": self.generate_pmm,
            "pmg": self.generate_pmg,
            "pgg": self.generate_pgg,
            "cmm": self.generate_cmm,
            "p4": self.generate_p4,
            "p4m": self.generate_p4m,
            "p4g": self.generate_p4g,
            "p3": self.generate_p3,
            "p3m1": self.generate_p3m1,
            "p31m": self.generate_p31m,
            "p6": self.generate_p6,
            "p6m": self.generate_p6m,
        }
        
        if group_name not in generators:
            raise ValueError(f"Unknown wallpaper group: {group_name}. "
                           f"Valid groups: {list(generators.keys())}")
        
        return generators[group_name](motif_size, **kwargs)
    
    def generate_all(self, 
                     motif_size: int = 64, 
                     **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate patterns for all 17 wallpaper groups.
        
        Returns:
            Dictionary mapping group names to pattern arrays
        """
        return {name: self.generate(name, motif_size, **kwargs) 
                for name in WALLPAPER_GROUPS.keys()}
    
    def get_group_info(self, group_name: str) -> WallpaperGroup:
        """Get information about a wallpaper group."""
        if group_name not in WALLPAPER_GROUPS:
            raise ValueError(f"Unknown group: {group_name}")
        return WALLPAPER_GROUPS[group_name]
    
    @staticmethod
    def list_groups() -> List[str]:
        """List all available wallpaper groups."""
        return list(WALLPAPER_GROUPS.keys())








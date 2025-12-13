#!/usr/bin/env python3
"""
Generate a colorful crystallographic patterns dataset.

Each wallpaper group has its own distinctive color palette,
making the patterns visually beautiful and easily distinguishable.

Usage:
    python scripts/generate_colored_dataset.py -n 500 -r 128
    python scripts/generate_colored_dataset.py -n 2000 -r 256 --output-dir ./data/colored
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image

from src.dataset.color_patterns import ColorPatternGenerator, GROUP_PALETTES
from src.dataset.pattern_generator import WALLPAPER_GROUPS
from src.dataset.pattern_generator_fixed import FixedWallpaperGenerator


class FixedColorPatternGenerator:
    """Color pattern generator using the mathematically correct symmetry generator."""
    
    def __init__(self, resolution: int = 256, seed: int = None):
        self.resolution = resolution
        self.base_generator = FixedWallpaperGenerator(resolution=resolution, seed=seed)
        self.rng = np.random.default_rng(seed)
    
    def _apply_colormap(self, pattern: np.ndarray, palette: dict) -> np.ndarray:
        """Apply color palette with sharp, well-defined colors."""
        h, w = pattern.shape
        colored = np.zeros((h, w, 3))
        
        # Normalize with contrast enhancement
        p_min, p_max = pattern.min(), pattern.max()
        if p_max > p_min:
            pattern = (pattern - p_min) / (p_max - p_min)
            # Enhance contrast with sigmoid-like curve
            pattern = 1 / (1 + np.exp(-6 * (pattern - 0.5)))
        
        bg = np.array(palette['bg'])
        primary = np.array(palette['primary'])
        secondary = np.array(palette['secondary'])
        accent = np.array(palette['accent'])
        
        # Sharp 4-band colormap for crisp definition
        for i in range(3):
            colored[:, :, i] = np.where(
                pattern < 0.25,
                bg[i] + (primary[i] - bg[i]) * (pattern / 0.25),
                np.where(
                    pattern < 0.5,
                    primary[i] + (secondary[i] - primary[i]) * ((pattern - 0.25) / 0.25),
                    np.where(
                        pattern < 0.75,
                        secondary[i] + (accent[i] - secondary[i]) * ((pattern - 0.5) / 0.25),
                        accent[i]
                    )
                )
            )
        
        return np.clip(colored, 0, 1)
    
    def generate(self, group_name: str, motif_size: int = 64, 
                 palette: dict = None, **kwargs) -> np.ndarray:
        """Generate colored pattern with correct symmetries."""
        # Generate grayscale with fixed generator
        gray_pattern = self.base_generator.generate(group_name, motif_size, **kwargs)
        
        # Apply color
        if palette is None:
            palette = GROUP_PALETTES.get(group_name, GROUP_PALETTES['p1'])
        
        return self._apply_colormap(gray_pattern, palette)
    
    def generate_random_palette(self) -> dict:
        """Generate random color palette."""
        import colorsys
        
        hue = self.rng.random()
        style = self.rng.choice(['complementary', 'analogous', 'triadic', 'split'])
        
        def hsv_to_rgb(h, s, v):
            return colorsys.hsv_to_rgb(h, s, v)
        
        bg_light = self.rng.random() > 0.5
        
        if style == 'complementary':
            if bg_light:
                bg = hsv_to_rgb(hue, 0.08, 0.95)
            else:
                bg = hsv_to_rgb(hue, 0.15, 0.1)
            primary = hsv_to_rgb(hue, 0.7, 0.7)
            secondary = hsv_to_rgb((hue + 0.5) % 1.0, 0.6, 0.6)
            accent = hsv_to_rgb((hue + 0.5) % 1.0, 0.8, 0.9)
        elif style == 'analogous':
            if bg_light:
                bg = hsv_to_rgb(hue, 0.05, 0.95)
            else:
                bg = hsv_to_rgb(hue, 0.15, 0.12)
            primary = hsv_to_rgb(hue, 0.7, 0.7)
            secondary = hsv_to_rgb((hue + 0.08) % 1.0, 0.6, 0.6)
            accent = hsv_to_rgb((hue - 0.08) % 1.0, 0.8, 0.9)
        elif style == 'triadic':
            bg = hsv_to_rgb(hue, 0.1, 0.12)
            primary = hsv_to_rgb(hue, 0.7, 0.75)
            secondary = hsv_to_rgb((hue + 0.33) % 1.0, 0.6, 0.65)
            accent = hsv_to_rgb((hue + 0.66) % 1.0, 0.8, 0.9)
        else:
            if bg_light:
                bg = hsv_to_rgb(hue, 0.03, 0.96)
            else:
                bg = hsv_to_rgb(hue, 0.2, 0.15)
            primary = hsv_to_rgb(hue, 0.75, 0.7)
            secondary = hsv_to_rgb((hue + 0.42) % 1.0, 0.55, 0.6)
            accent = hsv_to_rgb((hue + 0.58) % 1.0, 0.7, 0.85)
        
        return {'name': f'Random_{style}', 'bg': bg, 'primary': primary, 
                'secondary': secondary, 'accent': accent}


def generate_group_samples(args_tuple):
    """Generate samples for a single wallpaper group (for multiprocessing)."""
    (group_idx, group_name, num_samples, resolution, motif_sizes, 
     seed, save_images, output_path, random_colors) = args_tuple
    
    group_info = WALLPAPER_GROUPS[group_name]
    default_palette = GROUP_PALETTES[group_name]
    
    patterns = []
    labels = []
    metadata = []
    
    for sample_idx in range(num_samples):
        sample_seed = seed + group_idx * 10000 + sample_idx
        
        # Use FIXED generator for correct symmetries
        generator = FixedColorPatternGenerator(
            resolution=resolution,
            seed=sample_seed
        )
        
        motif_size = motif_sizes[sample_idx % len(motif_sizes)]
        # Higher complexity = more detail and sharper features
        complexity = (sample_idx % 8) + 3
        
        if random_colors:
            palette = generator.generate_random_palette()
        else:
            palette = default_palette
        
        pattern = generator.generate(
            group_name,
            motif_size=motif_size,
            complexity=complexity,
            palette=palette
        )
        
        patterns.append(pattern)
        labels.append(group_idx)
        metadata.append({
            'group_name': group_name,
            'palette_name': palette['name'],
            'lattice_type': group_info.lattice_type.value,
            'rotation_order': group_info.rotation_order,
            'has_reflection': group_info.has_reflection,
            'has_glide': group_info.has_glide,
            'motif_size': motif_size,
            'complexity': complexity,
            'seed': sample_seed
        })
        
        if save_images:
            img_path = output_path / "images" / group_name / f"{sample_idx:05d}.png"
            pattern_uint8 = (pattern * 255).astype(np.uint8)
            Image.fromarray(pattern_uint8, mode='RGB').save(img_path)
    
    return group_name, patterns, labels, metadata


def generate_colored_dataset(
    output_dir: str,
    num_samples_per_group: int = 500,
    resolution: int = 128,
    motif_sizes: list = [16, 24, 32, 40],  # Smaller = more repetitions = sharper
    seed: int = 42,
    save_images: bool = True,
    save_hdf5: bool = True,
    random_colors: bool = False,
    hdf5_name: str = "crystallographic_patterns_colored.h5",
    num_workers: int = 1,
):
    """
    Generate the full colored dataset.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples_per_group: Number of samples per wallpaper group
        resolution: Output image resolution
        motif_sizes: List of motif sizes to vary
        seed: Random seed for reproducibility
        save_images: Whether to save individual PNG images
        save_hdf5: Whether to save HDF5 file
        random_colors: Use random palettes per image
        hdf5_name: Name for the HDF5 file
        num_workers: Number of parallel workers (default 1 = sequential)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    groups = list(WALLPAPER_GROUPS.keys())
    total_samples = num_samples_per_group * len(groups)
    
    print("=" * 60)
    print("GENERATING COLORED CRYSTALLOGRAPHIC DATASET")
    print("=" * 60)
    print(f"\nSamples per group: {num_samples_per_group}")
    print(f"Total samples: {total_samples}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Random colors: {'Yes (color-invariant)' if random_colors else 'No (class-specific palettes)'}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    # Storage for HDF5
    all_patterns = []
    all_labels = []
    all_metadata = []
    
    # Create image directories
    if save_images:
        for group_name in groups:
            group_dir = output_path / "images" / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for each group
    group_args = [
        (group_idx, group_name, num_samples_per_group, resolution, motif_sizes,
         seed, save_images, output_path, random_colors)
        for group_idx, group_name in enumerate(groups)
    ]
    
    if num_workers > 1:
        # Parallel generation
        print(f"\n🚀 Generating in parallel with {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_group_samples, group_args),
                total=len(groups),
                desc="Groups"
            ))
        
        # Collect results in order
        for group_name, patterns, labels, metadata in results:
            all_patterns.extend(patterns)
            all_labels.extend(labels)
            all_metadata.extend(metadata)
            print(f"  ✓ {group_name}: {len(patterns)} samples")
    else:
        # Sequential generation (original behavior)
        for args in group_args:
            group_idx, group_name = args[0], args[1]
            default_palette = GROUP_PALETTES[group_name]
            
            if random_colors:
                print(f"\n[{group_idx+1}/{len(groups)}] {group_name} - Random colors")
            else:
                print(f"\n[{group_idx+1}/{len(groups)}] {group_name} - {default_palette['name']}")
            
            _, patterns, labels, metadata = generate_group_samples(args)
            all_patterns.extend(patterns)
            all_labels.extend(labels)
            all_metadata.extend(metadata)
    
    # Save HDF5
    if save_hdf5:
        hdf5_path = output_path / hdf5_name
        print(f"\nSaving HDF5 dataset to {hdf5_path}...")
        
        with h5py.File(hdf5_path, 'w') as f:
            # Main data - RGB patterns
            patterns = np.array(all_patterns)  # Shape: (N, H, W, 3)
            labels = np.array(all_labels)
            
            f.create_dataset('patterns', data=patterns, compression='gzip', 
                           compression_opts=9, 
                           chunks=(min(100, len(patterns)), resolution, resolution, 3))
            f.create_dataset('labels', data=labels)
            
            # Group names mapping
            group_names = np.array([g.encode('utf-8') for g in groups])
            f.create_dataset('group_names', data=group_names)
            
            # Palette names
            palette_names = np.array([GROUP_PALETTES[g]['name'].encode('utf-8') for g in groups])
            f.create_dataset('palette_names', data=palette_names)
            
            # Metadata
            f.attrs['num_samples_per_group'] = num_samples_per_group
            f.attrs['resolution'] = resolution
            f.attrs['num_groups'] = len(groups)
            f.attrs['total_samples'] = total_samples
            f.attrs['seed'] = seed
            f.attrs['channels'] = 3  # RGB
            
            # Metadata for each sample
            meta_grp = f.create_group('metadata')
            meta_grp.create_dataset('motif_size', 
                                   data=np.array([m['motif_size'] for m in all_metadata]))
            meta_grp.create_dataset('complexity',
                                   data=np.array([m['complexity'] for m in all_metadata]))
        
        file_size_mb = hdf5_path.stat().st_size / 1e6
        print(f"Dataset saved: {patterns.shape[0]} samples, {file_size_mb:.1f} MB")
    
    # Save split indices for train/val/test
    indices = np.arange(total_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    splits = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:]
    }
    
    np.savez(output_path / "splits.npz", **splits)
    print(f"\nData splits: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Format: RGB ({resolution}x{resolution}x3)")
    print(f"Output directory: {output_path}")
    
    if save_images:
        print(f"Images: {output_path / 'images'}")
    if save_hdf5:
        print(f"HDF5: {hdf5_path}")
    
    # Print palette preview
    print("\n🎨 Color Palettes Used:")
    for group_name in groups:
        palette = GROUP_PALETTES[group_name]
        print(f"  {group_name:6s} → {palette['name']}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate colored crystallographic patterns dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/colored_crystallographic",
        help="Output directory for the dataset"
    )
    
    parser.add_argument(
        "--samples-per-group", "-n",
        type=int,
        default=500,
        help="Number of samples per wallpaper group"
    )
    
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=128,
        help="Output image resolution (square)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't save individual PNG images"
    )
    
    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="Don't save HDF5 file"
    )
    
    parser.add_argument(
        "--random-colors",
        action="store_true",
        help="Use random color palettes per image instead of class-specific palettes"
    )
    
    parser.add_argument(
        "--hdf5-name",
        type=str,
        default=None,
        help="Custom name for the HDF5 file (default: auto-generated based on options)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (use 17 for one per group)"
    )
    
    args = parser.parse_args()
    
    # Auto-generate HDF5 name if not specified
    if args.hdf5_name is None:
        if args.random_colors:
            args.hdf5_name = f"crystallographic_random_colors_{args.resolution}.h5"
        else:
            args.hdf5_name = f"crystallographic_patterns_colored_{args.resolution}.h5"
    
    generate_colored_dataset(
        output_dir=args.output_dir,
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        seed=args.seed,
        save_images=not args.no_images,
        save_hdf5=not args.no_hdf5,
        random_colors=args.random_colors,
        hdf5_name=args.hdf5_name,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()



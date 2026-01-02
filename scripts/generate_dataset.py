#!/usr/bin/env python3
"""
Script to generate the crystallographic patterns dataset.

This script generates patterns for all 17 wallpaper groups and saves them
in various formats suitable for training neural networks.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image

from src.dataset.pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS


def generate_dataset(
    output_dir: str,
    num_samples_per_group: int = 500,
    resolution: int = 128,
    motif_sizes: list = [24, 32, 48, 64],
    seed: int = 42,
    save_images: bool = True,
    save_hdf5: bool = True,
):
    """
    Generate the full dataset.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples_per_group: Number of samples per wallpaper group
        resolution: Output image resolution
        motif_sizes: List of motif sizes to vary
        seed: Random seed for reproducibility
        save_images: Whether to save individual PNG images
        save_hdf5: Whether to save HDF5 file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    groups = list(WALLPAPER_GROUPS.keys())
    total_samples = num_samples_per_group * len(groups)
    
    print(f"Generating {total_samples} samples ({num_samples_per_group} per group)")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Output directory: {output_path}")
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
    
    # Generate samples
    for group_idx, group_name in enumerate(groups):
        group_info = WALLPAPER_GROUPS[group_name]
        print(f"\n[{group_idx+1}/{len(groups)}] Generating {group_name} "
              f"({group_info.lattice_type.value}, {group_info.rotation_order}-fold rotation)")
        
        for sample_idx in tqdm(range(num_samples_per_group), desc=f"  {group_name}"):
            # Deterministic seed for reproducibility
            sample_seed = seed + group_idx * 10000 + sample_idx
            
            generator = WallpaperGroupGenerator(
                resolution=resolution,
                seed=sample_seed
            )
            
            # Vary parameters
            motif_size = motif_sizes[sample_idx % len(motif_sizes)]
            complexity = (sample_idx % 6) + 2  # 2 to 7
            motif_types = ["gaussian", "geometric", "mixed"]
            motif_type = motif_types[sample_idx % len(motif_types)]
            
            # Generate pattern
            pattern = generator.generate(
                group_name,
                motif_size=motif_size,
                complexity=complexity,
                motif_type=motif_type
            )
            
            # Store
            all_patterns.append(pattern)
            all_labels.append(group_idx)
            all_metadata.append({
                'group_name': group_name,
                'lattice_type': group_info.lattice_type.value,
                'rotation_order': group_info.rotation_order,
                'has_reflection': group_info.has_reflection,
                'has_glide': group_info.has_glide,
                'motif_size': motif_size,
                'complexity': complexity,
                'motif_type': motif_type,
                'seed': sample_seed
            })
            
            # Save individual image
            if save_images:
                img_path = output_path / "images" / group_name / f"{sample_idx:05d}.png"
                # Normalize and convert to 8-bit
                pattern_norm = (pattern * 255).astype(np.uint8)
                Image.fromarray(pattern_norm).save(img_path)
    
    # Save HDF5
    if save_hdf5:
        hdf5_path = output_path / "crystallographic_patterns.h5"
        print(f"\nSaving HDF5 dataset to {hdf5_path}...")
        
        with h5py.File(hdf5_path, 'w') as f:
            # Main data
            patterns = np.array(all_patterns)
            labels = np.array(all_labels)
            
            f.create_dataset('patterns', data=patterns, compression='gzip', 
                           compression_opts=9, chunks=(100, resolution, resolution))
            f.create_dataset('labels', data=labels)
            
            # Group names mapping
            group_names = np.array([g.encode('utf-8') for g in groups])
            f.create_dataset('group_names', data=group_names)
            
            # Metadata
            f.attrs['num_samples_per_group'] = num_samples_per_group
            f.attrs['resolution'] = resolution
            f.attrs['num_groups'] = len(groups)
            f.attrs['total_samples'] = total_samples
            f.attrs['seed'] = seed
            
            # Metadata for each sample
            meta_grp = f.create_group('metadata')
            meta_grp.create_dataset('motif_size', 
                                   data=np.array([m['motif_size'] for m in all_metadata]))
            meta_grp.create_dataset('complexity',
                                   data=np.array([m['complexity'] for m in all_metadata]))
            meta_grp.create_dataset('motif_type',
                                   data=np.array([m['motif_type'].encode('utf-8') for m in all_metadata]))
        
        print(f"Dataset saved: {patterns.shape[0]} samples, "
              f"{patterns.nbytes / 1e6:.1f} MB")
    
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
    print(f"\nData splits saved: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Samples per group: {num_samples_per_group}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Output directory: {output_path}")
    
    if save_images:
        print(f"Images saved to: {output_path / 'images'}")
    if save_hdf5:
        print(f"HDF5 file: {hdf5_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate crystallographic patterns dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/crystallographic",
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
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        seed=args.seed,
        save_images=not args.no_images,
        save_hdf5=not args.no_hdf5,
    )


if __name__ == "__main__":
    main()









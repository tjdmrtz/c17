#!/usr/bin/env python3
"""
Verify that images in the crystallographic dataset actually have 
the symmetry of the group they belong to.

For each wallpaper group, we apply its symmetry transformations
and check if the image is (approximately) invariant.

Usage:
    python scripts/verify_dataset_symmetry.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import rotate as scipy_rotate
from tqdm import tqdm
import json


# All 17 wallpaper groups
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array, convert to grayscale float [0,1]."""
    img = Image.open(path)
    arr = np.array(img).astype(np.float32) / 255.0
    
    # Convert to grayscale if color
    if arr.ndim == 3:
        # Use luminance formula
        arr = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    
    return arr


def rotate_90(img: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate by 90*k degrees."""
    return np.rot90(img, k)


def rotate_arbitrary(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by arbitrary angle with periodic boundary approximation."""
    # Pad image to handle rotation, then crop back
    h, w = img.shape
    
    # Tile 3x3 to simulate periodic boundaries
    tiled = np.tile(img, (3, 3))
    
    # Rotate the tiled image
    rotated = scipy_rotate(tiled, angle, reshape=False, order=1, mode='wrap')
    
    # Extract center
    return rotated[h:2*h, w:2*w]


def flip_h(img: np.ndarray) -> np.ndarray:
    """Horizontal flip (left-right)."""
    return np.fliplr(img)


def flip_v(img: np.ndarray) -> np.ndarray:
    """Vertical flip (up-down)."""
    return np.flipud(img)


def roll_half(img: np.ndarray, axis: int) -> np.ndarray:
    """Roll by half the size along axis."""
    return np.roll(img, img.shape[axis] // 2, axis=axis)


def get_symmetry_transforms(group: str) -> list:
    """
    Get list of (name, transform_function) for a wallpaper group.
    
    Each transform should leave the pattern invariant if it has the correct symmetry.
    """
    transforms = {
        'p1': [
            # p1 has only identity - no additional symmetry to test
        ],
        'p2': [
            ('rot180', lambda x: rotate_90(x, 2)),
        ],
        'pm': [
            ('flip_h', flip_h),
        ],
        'pg': [
            ('glide_x', lambda x: roll_half(flip_h(x), axis=1)),
        ],
        'cm': [
            ('flip_h', flip_h),
        ],
        'pmm': [
            ('flip_h', flip_h),
            ('flip_v', flip_v),
            ('rot180', lambda x: rotate_90(x, 2)),
        ],
        'pmg': [
            ('flip_h', flip_h),
            ('rot180', lambda x: rotate_90(x, 2)),
        ],
        'pgg': [
            ('rot180', lambda x: rotate_90(x, 2)),
            ('glide_x', lambda x: roll_half(flip_h(x), axis=1)),
        ],
        'cmm': [
            ('flip_h', flip_h),
            ('flip_v', flip_v),
            ('rot180', lambda x: rotate_90(x, 2)),
        ],
        'p4': [
            ('rot90', lambda x: rotate_90(x, 1)),
            ('rot180', lambda x: rotate_90(x, 2)),
            ('rot270', lambda x: rotate_90(x, 3)),
        ],
        'p4m': [
            ('rot90', lambda x: rotate_90(x, 1)),
            ('rot180', lambda x: rotate_90(x, 2)),
            ('rot270', lambda x: rotate_90(x, 3)),
            ('flip_h', flip_h),
            ('flip_v', flip_v),
        ],
        'p4g': [
            ('rot90', lambda x: rotate_90(x, 1)),
            ('rot180', lambda x: rotate_90(x, 2)),
            ('rot270', lambda x: rotate_90(x, 3)),
        ],
        'p3': [
            ('rot120', lambda x: rotate_arbitrary(x, 120)),
            ('rot240', lambda x: rotate_arbitrary(x, 240)),
        ],
        'p3m1': [
            ('rot120', lambda x: rotate_arbitrary(x, 120)),
            ('rot240', lambda x: rotate_arbitrary(x, 240)),
            ('flip_h', flip_h),
        ],
        'p31m': [
            ('rot120', lambda x: rotate_arbitrary(x, 120)),
            ('rot240', lambda x: rotate_arbitrary(x, 240)),
            ('flip_v', flip_v),
        ],
        'p6': [
            ('rot60', lambda x: rotate_arbitrary(x, 60)),
            ('rot120', lambda x: rotate_arbitrary(x, 120)),
            ('rot180', lambda x: rotate_90(x, 2)),
            ('rot240', lambda x: rotate_arbitrary(x, 240)),
            ('rot300', lambda x: rotate_arbitrary(x, 300)),
        ],
        'p6m': [
            ('rot60', lambda x: rotate_arbitrary(x, 60)),
            ('rot120', lambda x: rotate_arbitrary(x, 120)),
            ('rot180', lambda x: rotate_90(x, 2)),
            ('rot240', lambda x: rotate_arbitrary(x, 240)),
            ('rot300', lambda x: rotate_arbitrary(x, 300)),
            ('flip_h', flip_h),
            ('flip_v', flip_v),
        ],
    }
    
    return transforms.get(group, [])


def compute_symmetry_error(img: np.ndarray, group: str) -> dict:
    """
    Compute symmetry error for an image.
    
    Returns dict with error for each transformation.
    Lower error = better symmetry.
    """
    transforms = get_symmetry_transforms(group)
    
    if not transforms:
        return {'no_transforms': 0.0}
    
    errors = {}
    for name, transform in transforms:
        transformed = transform(img)
        
        # Mean absolute error between original and transformed
        error = np.abs(img - transformed).mean()
        errors[name] = float(error)
    
    return errors


def verify_group(data_dir: Path, group: str, n_samples: int = 50) -> dict:
    """
    Verify symmetry of images in a group.
    
    Returns statistics about symmetry errors.
    """
    group_dir = data_dir / group
    
    if not group_dir.exists():
        return {'error': f'Directory not found: {group_dir}'}
    
    image_files = sorted(group_dir.glob('*.png'))[:n_samples]
    
    if not image_files:
        return {'error': 'No images found'}
    
    all_errors = []
    per_transform_errors = {}
    
    for img_path in image_files:
        img = load_image(img_path)
        errors = compute_symmetry_error(img, group)
        
        for name, error in errors.items():
            if name not in per_transform_errors:
                per_transform_errors[name] = []
            per_transform_errors[name].append(error)
        
        # Average error across all transforms
        avg_error = np.mean(list(errors.values()))
        all_errors.append(avg_error)
    
    # Compute statistics
    result = {
        'group': group,
        'n_images': len(image_files),
        'mean_error': float(np.mean(all_errors)),
        'std_error': float(np.std(all_errors)),
        'max_error': float(np.max(all_errors)),
        'min_error': float(np.min(all_errors)),
        'per_transform': {}
    }
    
    for name, errors in per_transform_errors.items():
        result['per_transform'][name] = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
        }
    
    return result


def visualize_symmetry_check(data_dir: Path, group: str, save_path: Path, n_examples: int = 3):
    """
    Visualize symmetry transformations on example images.
    """
    group_dir = data_dir / group
    image_files = sorted(group_dir.glob('*.png'))[:n_examples]
    
    transforms = get_symmetry_transforms(group)
    
    if not transforms:
        print(f"  {group}: No transforms to visualize (p1 has only identity)")
        return
    
    n_transforms = len(transforms)
    
    fig, axes = plt.subplots(n_examples, n_transforms + 2, 
                             figsize=((n_transforms + 2) * 2, n_examples * 2))
    
    if n_examples == 1:
        axes = [axes]
    
    fig.patch.set_facecolor('#0a0a0a')
    
    for row, img_path in enumerate(image_files):
        img = load_image(img_path)
        
        # Original
        axes[row][0].imshow(img, cmap='viridis')
        axes[row][0].set_title('Original', fontsize=10, color='white')
        axes[row][0].axis('off')
        
        # Each transform
        for col, (name, transform) in enumerate(transforms, 1):
            transformed = transform(img)
            diff = np.abs(img - transformed)
            error = diff.mean()
            
            axes[row][col].imshow(transformed, cmap='viridis')
            axes[row][col].set_title(f'{name}\nErr={error:.4f}', 
                                     fontsize=9, color='white')
            axes[row][col].axis('off')
        
        # Difference map (average of all transforms)
        all_diffs = []
        for name, transform in transforms:
            transformed = transform(img)
            all_diffs.append(np.abs(img - transformed))
        avg_diff = np.mean(all_diffs, axis=0)
        
        axes[row][-1].imshow(avg_diff, cmap='hot', vmin=0, vmax=0.2)
        axes[row][-1].set_title(f'Avg Diff\n{avg_diff.mean():.4f}', 
                                fontsize=9, color='white')
        axes[row][-1].axis('off')
    
    plt.suptitle(f'Symmetry Verification: {group}', 
                 fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()


def main():
    data_dir = Path('data/colored_crystallographic/images')
    output_dir = Path('output/symmetry_verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SYMMETRY VERIFICATION OF CRYSTALLOGRAPHIC DATASET")
    print("="*70)
    print(f"Dataset: {data_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Verify all groups
    results = {}
    
    for group in tqdm(ALL_17_GROUPS, desc="Verifying groups"):
        result = verify_group(data_dir, group, n_samples=100)
        results[group] = result
        
        # Create visualization for this group
        visualize_symmetry_check(data_dir, group, 
                                 output_dir / f'verify_{group}.png', n_examples=3)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Group':<8} {'Mean Err':<12} {'Std':<10} {'Status':<10}")
    print("-"*50)
    
    passed = 0
    failed = 0
    threshold = 0.05  # 5% error threshold
    
    for group in ALL_17_GROUPS:
        r = results[group]
        if 'error' in r:
            print(f"{group:<8} {r['error']}")
            failed += 1
        else:
            status = "✓ PASS" if r['mean_error'] < threshold else "✗ FAIL"
            color_status = status
            print(f"{group:<8} {r['mean_error']:<12.4f} {r['std_error']:<10.4f} {status}")
            
            if r['mean_error'] < threshold:
                passed += 1
            else:
                failed += 1
    
    print("-"*50)
    print(f"PASSED: {passed}/17 groups")
    print(f"FAILED: {failed}/17 groups")
    
    # Detailed per-transform analysis
    print("\n" + "="*70)
    print("PER-TRANSFORM ANALYSIS")
    print("="*70)
    
    for group in ALL_17_GROUPS:
        r = results[group]
        if 'per_transform' in r and r['per_transform']:
            print(f"\n{group}:")
            for name, stats in r['per_transform'].items():
                status = "✓" if stats['mean'] < threshold else "✗"
                print(f"  {name:<12}: mean={stats['mean']:.4f} std={stats['std']:.4f} {status}")
    
    # Save results to JSON
    with open(output_dir / 'verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'verification_results.json'}")
    
    # Create summary visualization
    create_summary_plot(results, output_dir / 'verification_summary.png')
    
    print(f"Summary plot saved to {output_dir / 'verification_summary.png'}")


def create_summary_plot(results: dict, save_path: Path):
    """Create a bar chart summarizing symmetry errors for all groups."""
    
    groups = []
    errors = []
    stds = []
    
    for group in ALL_17_GROUPS:
        r = results[group]
        if 'mean_error' in r:
            groups.append(group)
            errors.append(r['mean_error'])
            stds.append(r['std_error'])
    
    # Color by lattice type
    lattice_colors = {
        'p1': '#2980b9', 'p2': '#2980b9',
        'pm': '#27ae60', 'pg': '#27ae60', 'cm': '#27ae60',
        'pmm': '#27ae60', 'pmg': '#27ae60', 'pgg': '#27ae60', 'cmm': '#27ae60',
        'p4': '#8e44ad', 'p4m': '#8e44ad', 'p4g': '#8e44ad',
        'p3': '#e74c3c', 'p3m1': '#e74c3c', 'p31m': '#e74c3c',
        'p6': '#e74c3c', 'p6m': '#e74c3c',
    }
    
    colors = [lattice_colors.get(g, 'gray') for g in groups]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    bars = ax.bar(groups, errors, yerr=stds, color=colors, 
                  edgecolor='white', linewidth=0.5, capsize=3, alpha=0.8)
    
    # Threshold line
    ax.axhline(y=0.05, color='#ff6b6b', linestyle='--', linewidth=2, label='Threshold (5%)')
    
    ax.set_xlabel('Wallpaper Group', fontsize=12, color='white')
    ax.set_ylabel('Mean Symmetry Error', fontsize=12, color='white')
    ax.set_title('Symmetry Verification: Mean Error per Group\n'
                 '(Lower = Better Symmetry)', fontsize=14, color='white', fontweight='bold')
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    
    # Add pass/fail labels
    for i, (bar, err) in enumerate(zip(bars, errors)):
        label = '✓' if err < 0.05 else '✗'
        color = '#2ecc71' if err < 0.05 else '#e74c3c'
        ax.text(bar.get_x() + bar.get_width()/2, err + stds[i] + 0.01, 
                label, ha='center', va='bottom', fontsize=12, color=color, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()






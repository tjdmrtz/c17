#!/usr/bin/env python3
"""
Verify symmetry properties of crystallographic patterns and remove invalid ones.

This script:
1. Loads all patterns from the H5 dataset
2. Verifies each pattern has the correct symmetry for its assigned group
3. Reports invalid patterns
4. Creates a cleaned dataset

Uses multiprocessing with 17 cores (one per group).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from tqdm import tqdm

# Symmetry tolerance (patterns are pixelated, so we need some tolerance)
# Higher value = more lenient
TOLERANCE = 0.15  # 15% pixel difference allowed


ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]


def rotate_90(img: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees counterclockwise."""
    return np.rot90(img, k=1, axes=(0, 1))


def rotate_180(img: np.ndarray) -> np.ndarray:
    """Rotate image 180 degrees."""
    return np.rot90(img, k=2, axes=(0, 1))


def rotate_270(img: np.ndarray) -> np.ndarray:
    """Rotate image 270 degrees counterclockwise."""
    return np.rot90(img, k=3, axes=(0, 1))


def flip_h(img: np.ndarray) -> np.ndarray:
    """Flip horizontally."""
    return np.flip(img, axis=1)


def flip_v(img: np.ndarray) -> np.ndarray:
    """Flip vertically."""
    return np.flip(img, axis=0)


def rotate_120(img: np.ndarray) -> np.ndarray:
    """Rotate 120 degrees using scipy."""
    from scipy.ndimage import rotate
    return rotate(img, 120, axes=(0, 1), reshape=False, mode='wrap')


def rotate_60(img: np.ndarray) -> np.ndarray:
    """Rotate 60 degrees using scipy."""
    from scipy.ndimage import rotate
    return rotate(img, 60, axes=(0, 1), reshape=False, mode='wrap')


def translate(img: np.ndarray, shift_h: int, shift_v: int) -> np.ndarray:
    """Translate with periodic boundary."""
    return np.roll(np.roll(img, shift_h, axis=1), shift_v, axis=0)


def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute similarity between two images.
    Returns value in [0, 1] where 1 = identical.
    """
    # Normalize both images
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0
    
    # Mean absolute error
    mae = np.abs(img1 - img2).mean()
    
    # Convert to similarity
    similarity = 1.0 - mae
    
    return similarity


def check_symmetry(img: np.ndarray, transform_func, tolerance: float = TOLERANCE) -> Tuple[bool, float]:
    """
    Check if image has a specific symmetry.
    
    Returns:
        (is_valid, similarity_score)
    """
    transformed = transform_func(img)
    similarity = compute_similarity(img, transformed)
    is_valid = similarity >= (1.0 - tolerance)
    return is_valid, similarity


def verify_group_symmetry(img: np.ndarray, group_name: str, tolerance: float = TOLERANCE) -> Tuple[bool, Dict[str, float]]:
    """
    Verify that an image has the symmetry properties of a specific group.
    
    Returns:
        (is_valid, scores_dict)
    """
    h, w = img.shape[:2]
    scores = {}
    
    if group_name == 'p1':
        # p1 has no point symmetry, only translation
        # We can't really verify this, so we accept all
        return True, {'translation': 1.0}
    
    elif group_name == 'p2':
        # 2-fold rotation (180°)
        valid, score = check_symmetry(img, rotate_180, tolerance)
        scores['rot180'] = score
        return valid, scores
    
    elif group_name == 'pm':
        # Reflection (horizontal or vertical)
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        scores['flip_h'] = score_h
        scores['flip_v'] = score_v
        return valid_h or valid_v, scores
    
    elif group_name == 'pg':
        # Glide reflection - harder to verify exactly
        # Check for approximate glide: flip + translate
        def glide_h(x):
            return translate(flip_v(x), h // 2, 0)
        def glide_v(x):
            return translate(flip_h(x), 0, w // 2)
        
        valid1, score1 = check_symmetry(img, glide_h, tolerance * 1.5)  # More lenient
        valid2, score2 = check_symmetry(img, glide_v, tolerance * 1.5)
        scores['glide_h'] = score1
        scores['glide_v'] = score2
        return valid1 or valid2, scores
    
    elif group_name == 'cm':
        # Reflection with centered cell - check reflection
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        scores['flip_h'] = score_h
        scores['flip_v'] = score_v
        return valid_h or valid_v, scores
    
    elif group_name == 'pmm':
        # Two perpendicular reflections
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['flip_h'] = score_h
        scores['flip_v'] = score_v
        scores['rot180'] = score_180
        # Should have at least 2-fold rotation from the two reflections
        return (valid_h and valid_v) or (valid_180 and (valid_h or valid_v)), scores
    
    elif group_name == 'pmg':
        # Reflection + glide
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['flip_h'] = score_h
        scores['flip_v'] = score_v
        scores['rot180'] = score_180
        return valid_180 and (valid_h or valid_v), scores
    
    elif group_name == 'pgg':
        # Two glide reflections + 2-fold rotation
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['rot180'] = score_180
        return valid_180, scores  # At minimum should have 180° rotation
    
    elif group_name == 'cmm':
        # Two reflections + 2-fold rotation
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['flip_h'] = score_h
        scores['flip_v'] = score_v
        scores['rot180'] = score_180
        return valid_180 or (valid_h and valid_v), scores
    
    elif group_name == 'p4':
        # 4-fold rotation
        valid_90, score_90 = check_symmetry(img, rotate_90, tolerance)
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['rot90'] = score_90
        scores['rot180'] = score_180
        return valid_90 and valid_180, scores
    
    elif group_name == 'p4m':
        # 4-fold rotation + reflections
        valid_90, score_90 = check_symmetry(img, rotate_90, tolerance)
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        scores['rot90'] = score_90
        scores['flip_h'] = score_h
        scores['flip_v'] = score_v
        return valid_90 and (valid_h or valid_v), scores
    
    elif group_name == 'p4g':
        # 4-fold rotation + glide reflections
        valid_90, score_90 = check_symmetry(img, rotate_90, tolerance)
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['rot90'] = score_90
        scores['rot180'] = score_180
        return valid_90, scores
    
    elif group_name == 'p3':
        # 3-fold rotation
        valid_120, score_120 = check_symmetry(img, rotate_120, tolerance * 1.5)  # More lenient for non-90° rotations
        scores['rot120'] = score_120
        return valid_120, scores
    
    elif group_name == 'p3m1':
        # 3-fold rotation + reflections
        valid_120, score_120 = check_symmetry(img, rotate_120, tolerance * 1.5)
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        scores['rot120'] = score_120
        scores['flip_h'] = score_h
        return valid_120, scores  # Focus on rotation
    
    elif group_name == 'p31m':
        # 3-fold rotation + different reflections
        valid_120, score_120 = check_symmetry(img, rotate_120, tolerance * 1.5)
        valid_v, score_v = check_symmetry(img, flip_v, tolerance)
        scores['rot120'] = score_120
        scores['flip_v'] = score_v
        return valid_120, scores  # Focus on rotation
    
    elif group_name == 'p6':
        # 6-fold rotation
        valid_60, score_60 = check_symmetry(img, rotate_60, tolerance * 1.5)
        valid_120, score_120 = check_symmetry(img, rotate_120, tolerance * 1.5)
        valid_180, score_180 = check_symmetry(img, rotate_180, tolerance)
        scores['rot60'] = score_60
        scores['rot120'] = score_120
        scores['rot180'] = score_180
        return valid_120 and valid_180, scores  # Should have at least 3-fold and 2-fold
    
    elif group_name == 'p6m':
        # 6-fold rotation + reflections
        valid_60, score_60 = check_symmetry(img, rotate_60, tolerance * 1.5)
        valid_120, score_120 = check_symmetry(img, rotate_120, tolerance * 1.5)
        valid_h, score_h = check_symmetry(img, flip_h, tolerance)
        scores['rot60'] = score_60
        scores['rot120'] = score_120
        scores['flip_h'] = score_h
        return valid_120, scores  # Focus on 3-fold
    
    # Unknown group
    return True, {}


def verify_group_patterns(args: Tuple[int, str, np.ndarray, np.ndarray, float]) -> Dict:
    """
    Verify all patterns for a single group.
    
    Args:
        args: (group_idx, group_name, patterns, indices, tolerance)
        
    Returns:
        Dict with results
    """
    group_idx, group_name, patterns, indices, tolerance = args
    
    valid_indices = []
    invalid_indices = []
    scores_list = []
    
    for i, (pattern, idx) in enumerate(zip(patterns, indices)):
        is_valid, scores = verify_group_symmetry(pattern, group_name, tolerance)
        
        if is_valid:
            valid_indices.append(int(idx))
        else:
            invalid_indices.append({
                'index': int(idx),
                'scores': {k: float(v) for k, v in scores.items()},
                'reason': f"Failed symmetry check for {group_name}"
            })
        
        scores_list.append(scores)
    
    # Calculate average scores
    avg_scores = {}
    if scores_list:
        all_keys = set()
        for s in scores_list:
            all_keys.update(s.keys())
        for key in all_keys:
            values = [s.get(key, 0) for s in scores_list if key in s]
            if values:
                avg_scores[key] = float(np.mean(values))
    
    return {
        'group_idx': group_idx,
        'group_name': group_name,
        'total': len(patterns),
        'valid': len(valid_indices),
        'invalid': len(invalid_indices),
        'valid_indices': valid_indices,
        'invalid_details': invalid_indices,
        'avg_scores': avg_scores,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify and clean crystallographic dataset')
    parser.add_argument('--h5-path', type=str,
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5',
                       help='Path to H5 dataset')
    parser.add_argument('--tolerance', type=float, default=TOLERANCE,
                       help='Symmetry tolerance (0-1, higher = more lenient)')
    parser.add_argument('--workers', type=int, default=17,
                       help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='data/colored_crystallographic',
                       help='Output directory for cleaned dataset')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only report, do not modify dataset')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SYMMETRY VERIFICATION FOR CRYSTALLOGRAPHIC DATASET")
    print("=" * 70)
    print(f"\nDataset: {args.h5_path}")
    print(f"Tolerance: {args.tolerance} ({args.tolerance*100:.0f}% pixel difference allowed)")
    print(f"Workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    
    # Load dataset
    print("\nLoading dataset...")
    with h5py.File(args.h5_path, 'r') as f:
        patterns = f['patterns'][:]  # Load all into memory
        labels = f['labels'][:]
        group_names = [g.decode() for g in f['group_names'][:]]
    
    print(f"Loaded {len(patterns)} patterns")
    print(f"Shape: {patterns.shape}")
    
    # Normalize if needed
    if patterns.max() > 1:
        patterns = patterns / 255.0
    
    # Prepare data for each group
    group_data = []
    for group_idx, group_name in enumerate(ALL_17_GROUPS):
        mask = labels == group_idx
        indices = np.where(mask)[0]
        group_patterns = patterns[mask]
        
        if len(group_patterns) > 0:
            group_data.append((group_idx, group_name, group_patterns, indices, args.tolerance))
            print(f"  {group_name}: {len(group_patterns)} patterns")
    
    # Process in parallel
    print(f"\nVerifying symmetry properties using {args.workers} workers...")
    print("-" * 70)
    
    results = []
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap(verify_group_patterns, group_data), 
                          total=len(group_data), desc="Processing groups"):
            results.append(result)
            
            # Print intermediate results
            r = result
            status = "✓" if r['invalid'] == 0 else "⚠"
            print(f"\n{status} {r['group_name']}: {r['valid']}/{r['total']} valid "
                  f"({r['invalid']} invalid)")
            if r['avg_scores']:
                scores_str = ", ".join([f"{k}={v:.3f}" for k, v in r['avg_scores'].items()])
                print(f"   Avg scores: {scores_str}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_valid = sum(r['valid'] for r in results)
    total_invalid = sum(r['invalid'] for r in results)
    total = sum(r['total'] for r in results)
    
    print(f"\nTotal patterns: {total}")
    print(f"Valid patterns: {total_valid} ({100*total_valid/total:.1f}%)")
    print(f"Invalid patterns: {total_invalid} ({100*total_invalid/total:.1f}%)")
    
    # Details of invalid patterns
    if total_invalid > 0:
        print("\n" + "-" * 70)
        print("INVALID PATTERNS BY GROUP")
        print("-" * 70)
        
        all_invalid_indices = []
        
        for r in results:
            if r['invalid'] > 0:
                print(f"\n{r['group_name']} ({r['invalid']} invalid):")
                for detail in r['invalid_details'][:5]:  # Show first 5
                    scores_str = ", ".join([f"{k}={v:.3f}" for k, v in detail['scores'].items()])
                    print(f"  Index {detail['index']}: {scores_str}")
                
                if r['invalid'] > 5:
                    print(f"  ... and {r['invalid'] - 5} more")
                
                all_invalid_indices.extend([d['index'] for d in r['invalid_details']])
        
        print(f"\nTotal invalid indices: {len(all_invalid_indices)}")
        
        if not args.dry_run and total_invalid > 0:
            print("\n" + "=" * 70)
            print("CREATING CLEANED DATASET")
            print("=" * 70)
            
            # Get valid indices
            all_valid_indices = []
            for r in results:
                all_valid_indices.extend(r['valid_indices'])
            all_valid_indices = sorted(all_valid_indices)
            
            print(f"\nKeeping {len(all_valid_indices)} valid patterns")
            
            # Create new splits
            output_dir = Path(args.output_dir)
            
            # Load original splits
            splits = np.load(output_dir / 'splits.npz')
            original_train = set(splits['train'])
            original_val = set(splits['val'])
            original_test = set(splits['test'])
            
            # Filter splits to only include valid indices
            valid_set = set(all_valid_indices)
            new_train = np.array([i for i in original_train if i in valid_set])
            new_val = np.array([i for i in original_val if i in valid_set])
            new_test = np.array([i for i in original_test if i in valid_set])
            
            print(f"\nNew splits:")
            print(f"  Train: {len(new_train)} (was {len(original_train)})")
            print(f"  Val: {len(new_val)} (was {len(original_val)})")
            print(f"  Test: {len(new_test)} (was {len(original_test)})")
            
            # Save new splits
            new_splits_path = output_dir / 'splits_cleaned.npz'
            np.savez(new_splits_path, train=new_train, val=new_val, test=new_test)
            print(f"\nSaved cleaned splits to: {new_splits_path}")
            
            # Save report
            report = {
                'tolerance': args.tolerance,
                'total_patterns': total,
                'valid_patterns': total_valid,
                'invalid_patterns': total_invalid,
                'invalid_indices': all_invalid_indices,
                'valid_indices': all_valid_indices,
                'by_group': [{
                    'group': r['group_name'],
                    'total': r['total'],
                    'valid': r['valid'],
                    'invalid': r['invalid'],
                    'avg_scores': r['avg_scores'],
                } for r in results]
            }
            
            report_path = output_dir / 'symmetry_verification_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Saved report to: {report_path}")
            
    else:
        print("\n✓ All patterns have valid symmetry!")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()




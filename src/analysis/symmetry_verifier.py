#!/usr/bin/env python3
"""
Symmetry Verifier for Crystallographic Patterns.

Verifies that generated patterns actually have the symmetries
claimed by their wallpaper group classification.

KEY INSIGHT: Wallpaper symmetries exist WITHIN the unit cell, not globally.
We must:
1. Detect the unit cell (period of repetition)
2. Extract a single unit cell
3. Verify symmetries WITHIN that cell

This version uses DIRECT PIXEL COMPARISON (no Fourier) for clarity.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import correlate2d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LatticeType(Enum):
    OBLIQUE = "oblique"
    RECTANGULAR = "rectangular"
    SQUARE = "square"
    HEXAGONAL = "hexagonal"


@dataclass
class SymmetryResult:
    """Result of a single symmetry check."""
    name: str
    present: bool
    score: float  # 0-1, higher = more symmetric
    threshold: float
    details: str = ""


@dataclass 
class GroupVerificationResult:
    """Complete verification result for a pattern."""
    expected_group: str
    verified: bool
    symmetry_scores: Dict[str, SymmetryResult]
    detected_period: Tuple[int, int]
    confidence: float
    suggested_group: Optional[str] = None
    message: str = ""


# Symmetry requirements for each wallpaper group
GROUP_PROPERTIES = {
    'p1':   {'rotation': 1, 'reflections': [], 'glides': []},
    'p2':   {'rotation': 2, 'reflections': [], 'glides': []},
    'pm':   {'rotation': 1, 'reflections': ['v'], 'glides': []},
    'pg':   {'rotation': 1, 'reflections': [], 'glides': ['v']},
    'cm':   {'rotation': 1, 'reflections': ['v'], 'glides': ['v']},
    'pmm':  {'rotation': 2, 'reflections': ['h', 'v'], 'glides': []},
    'pmg':  {'rotation': 2, 'reflections': ['v'], 'glides': ['h']},
    'pgg':  {'rotation': 2, 'reflections': [], 'glides': ['h', 'v']},
    'cmm':  {'rotation': 2, 'reflections': ['h', 'v'], 'glides': []},
    'p4':   {'rotation': 4, 'reflections': [], 'glides': []},
    'p4m':  {'rotation': 4, 'reflections': ['h', 'v', 'd1', 'd2'], 'glides': []},
    'p4g':  {'rotation': 4, 'reflections': ['d1', 'd2'], 'glides': ['h', 'v']},
    'p3':   {'rotation': 3, 'reflections': [], 'glides': []},
    'p3m1': {'rotation': 3, 'reflections': ['hex'], 'glides': []},
    'p31m': {'rotation': 3, 'reflections': ['hex'], 'glides': []},
    'p6':   {'rotation': 6, 'reflections': [], 'glides': []},
    'p6m':  {'rotation': 6, 'reflections': ['hex'], 'glides': []},
}


class SymmetryVerifier:
    """
    Verifies crystallographic symmetries in 2D patterns.
    
    This verifier works by:
    1. Detecting the unit cell period via autocorrelation
    2. Extracting a representative unit cell
    3. Checking symmetry operations WITHIN that cell
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 period_detection_method: str = 'autocorr'):
        """
        Initialize verifier.
        
        Args:
            similarity_threshold: Minimum similarity (0-1) to consider a symmetry present
            period_detection_method: 'autocorr' or 'known' (if period is provided)
        """
        self.threshold = similarity_threshold
        self.method = period_detection_method
    
    def verify(self, 
               image: np.ndarray, 
               expected_group: str,
               known_period: Optional[Tuple[int, int]] = None,
               verbose: bool = False) -> GroupVerificationResult:
        """
        Verify that an image has the expected wallpaper group symmetry.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            expected_group: Expected wallpaper group (e.g., 'p4m')
            known_period: If known, the (period_x, period_y) of the unit cell
            verbose: Print detailed info
            
        Returns:
            GroupVerificationResult
        """
        if expected_group not in GROUP_PROPERTIES:
            raise ValueError(f"Unknown group: {expected_group}")
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Normalize to [0, 1]
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Detect or use known period
        if known_period is not None:
            period_x, period_y = known_period
        else:
            period_x, period_y = self._detect_period(gray)
        
        if verbose:
            print(f"Unit cell period: ({period_x}, {period_y}) pixels")
        
        # Extract unit cell from center of image (avoid edge artifacts)
        cell = self._extract_unit_cell(gray, period_x, period_y)
        
        if verbose:
            print(f"Extracted cell shape: {cell.shape}")
        
        # Get required symmetries for this group
        props = GROUP_PROPERTIES[expected_group]
        
        # Check all required symmetries
        results = {}
        
        # 1. Rotation symmetry
        if props['rotation'] > 1:
            rot_result = self._check_rotation_symmetry(cell, props['rotation'])
            results['rotation'] = rot_result
            if verbose:
                status = "✓" if rot_result.present else "✗"
                print(f"  {status} {props['rotation']}-fold rotation: {rot_result.score:.3f}")
        
        # 2. Reflection symmetries
        for ref_type in props['reflections']:
            ref_result = self._check_reflection_symmetry(cell, ref_type)
            results[f'reflection_{ref_type}'] = ref_result
            if verbose:
                status = "✓" if ref_result.present else "✗"
                print(f"  {status} Reflection ({ref_type}): {ref_result.score:.3f}")
        
        # 3. Glide reflections
        for glide_type in props['glides']:
            glide_result = self._check_glide_symmetry(cell, glide_type)
            results[f'glide_{glide_type}'] = glide_result
            if verbose:
                status = "✓" if glide_result.present else "✗"
                print(f"  {status} Glide ({glide_type}): {glide_result.score:.3f}")
        
        # Calculate overall result
        if results:
            all_present = all(r.present for r in results.values())
            confidence = np.mean([r.score for r in results.values()])
        else:
            # p1 has no symmetries to check - always passes
            all_present = True
            confidence = 1.0
        
        # Check for unexpected higher symmetries
        unexpected = self._check_unexpected_symmetries(cell, expected_group)
        
        verified = all_present and not unexpected
        
        if verified:
            message = f"✓ Pattern verified as {expected_group}"
        else:
            missing = [k for k, v in results.items() if not v.present]
            if missing:
                message = f"✗ Missing symmetries: {missing}"
            elif unexpected:
                message = f"✗ Has unexpected symmetries (may be higher group)"
            else:
                message = "✗ Verification failed"
        
        return GroupVerificationResult(
            expected_group=expected_group,
            verified=verified,
            symmetry_scores=results,
            detected_period=(period_x, period_y),
            confidence=confidence,
            message=message
        )
    
    def _detect_period(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Detect unit cell period using autocorrelation.
        
        The autocorrelation of a periodic pattern has peaks at
        multiples of the period.
        """
        h, w = image.shape
        
        # Use only central portion to avoid edge effects
        margin = min(h, w) // 8
        center = image[margin:-margin, margin:-margin]
        
        # Compute autocorrelation directly (no FFT needed for explanation)
        # For efficiency, we use FFT-based autocorrelation
        from scipy.fft import fft2, ifft2, fftshift
        
        f = fft2(center)
        autocorr = np.real(ifft2(f * np.conj(f)))
        autocorr = fftshift(autocorr)
        autocorr = autocorr / autocorr.max()
        
        ch, cw = autocorr.shape
        center_y, center_x = ch // 2, cw // 2
        
        # Find peaks by looking for local maxima
        # Mask out the center peak
        min_dist = min(ch, cw) // 10
        
        # Look for the nearest significant peak
        best_period_x, best_period_y = cw // 4, ch // 4  # default
        best_score = 0
        
        for dy in range(min_dist, ch // 2):
            for dx in range(min_dist, cw // 2):
                # Check if this is a local maximum
                y, x = center_y + dy, center_x + dx
                if y >= ch or x >= cw:
                    continue
                    
                val = autocorr[y, x]
                if val > best_score and val > 0.3:  # Threshold for significance
                    # Check it's a local max
                    is_max = True
                    for ny in range(max(0, y-2), min(ch, y+3)):
                        for nx in range(max(0, x-2), min(cw, x+3)):
                            if autocorr[ny, nx] > val and (ny != y or nx != x):
                                is_max = False
                                break
                        if not is_max:
                            break
                    
                    if is_max:
                        best_score = val
                        best_period_x = dx
                        best_period_y = dy
        
        # Ensure reasonable bounds
        period_x = max(16, min(best_period_x, w // 2))
        period_y = max(16, min(best_period_y, h // 2))
        
        return period_x, period_y
    
    def _extract_unit_cell(self, 
                           image: np.ndarray, 
                           period_x: int, 
                           period_y: int) -> np.ndarray:
        """Extract a unit cell from the center of the image."""
        h, w = image.shape
        
        # Extract from center to avoid edge artifacts
        start_y = (h - period_y) // 2
        start_x = (w - period_x) // 2
        
        cell = image[start_y:start_y + period_y, start_x:start_x + period_x]
        return cell
    
    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity between two images using normalized correlation.
        
        Returns value in [0, 1] where 1 = identical.
        """
        # Ensure same shape
        if img1.shape != img2.shape:
            # Resize to match
            from scipy.ndimage import zoom
            target_shape = (min(img1.shape[0], img2.shape[0]),
                          min(img1.shape[1], img2.shape[1]))
            img1 = img1[:target_shape[0], :target_shape[1]]
            img2 = img2[:target_shape[0], :target_shape[1]]
        
        # Normalize
        img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)
        
        # Normalized correlation
        correlation = np.mean(img1_norm * img2_norm)
        
        # Map to [0, 1]
        similarity = (correlation + 1) / 2
        return float(np.clip(similarity, 0, 1))
    
    def _check_rotation_symmetry(self, 
                                  cell: np.ndarray, 
                                  order: int) -> SymmetryResult:
        """
        Check if cell has n-fold rotation symmetry.
        
        For n-fold symmetry, rotating by 360/n degrees should give the same image.
        """
        angle = 360.0 / order
        
        # For fair comparison, use central region (rotation affects corners)
        h, w = cell.shape
        margin = min(h, w) // 4
        
        scores = []
        for k in range(1, order):
            rot_angle = k * angle
            rotated = ndimage.rotate(cell, rot_angle, reshape=False, order=1, mode='wrap')
            
            # Compare central regions
            orig_center = cell[margin:-margin, margin:-margin] if margin > 0 else cell
            rot_center = rotated[margin:-margin, margin:-margin] if margin > 0 else rotated
            
            score = self._compute_similarity(orig_center, rot_center)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 1.0
        present = avg_score >= self.threshold
        
        return SymmetryResult(
            name=f"C{order} rotation",
            present=present,
            score=avg_score,
            threshold=self.threshold,
            details=f"Average similarity under {order}-fold rotation"
        )
    
    def _check_reflection_symmetry(self, 
                                    cell: np.ndarray, 
                                    axis_type: str) -> SymmetryResult:
        """
        Check reflection symmetry.
        
        axis_type:
            'h' = horizontal axis (flip vertically)
            'v' = vertical axis (flip horizontally)
            'd1' = diagonal (transpose)
            'd2' = anti-diagonal
            'hex' = hexagonal (60° axes)
        """
        if axis_type == 'h':
            reflected = np.flipud(cell)
        elif axis_type == 'v':
            reflected = np.fliplr(cell)
        elif axis_type == 'd1':
            reflected = cell.T
            if cell.shape[0] != cell.shape[1]:
                # Non-square, resize
                reflected = ndimage.zoom(reflected, 
                                        (cell.shape[0]/reflected.shape[0],
                                         cell.shape[1]/reflected.shape[1]))
        elif axis_type == 'd2':
            reflected = np.flipud(np.fliplr(cell.T))
            if cell.shape[0] != cell.shape[1]:
                reflected = ndimage.zoom(reflected,
                                        (cell.shape[0]/reflected.shape[0],
                                         cell.shape[1]/reflected.shape[1]))
        elif axis_type == 'hex':
            # For hexagonal, check 60° reflection
            rotated = ndimage.rotate(cell, 60, reshape=False, order=1, mode='wrap')
            reflected = np.flipud(rotated)
        else:
            reflected = cell  # Unknown type
        
        score = self._compute_similarity(cell, reflected)
        present = score >= self.threshold
        
        return SymmetryResult(
            name=f"Reflection ({axis_type})",
            present=present,
            score=score,
            threshold=self.threshold,
            details=f"Mirror symmetry across {axis_type} axis"
        )
    
    def _check_glide_symmetry(self, 
                               cell: np.ndarray, 
                               axis_type: str) -> SymmetryResult:
        """
        Check glide reflection symmetry.
        
        Glide = reflection + translation by half the period.
        """
        h, w = cell.shape
        
        if axis_type == 'h':
            # Reflect vertically, translate horizontally
            reflected = np.flipud(cell)
            glided = np.roll(reflected, w // 2, axis=1)
        elif axis_type == 'v':
            # Reflect horizontally, translate vertically
            reflected = np.fliplr(cell)
            glided = np.roll(reflected, h // 2, axis=0)
        else:
            glided = cell
        
        score = self._compute_similarity(cell, glided)
        present = score >= self.threshold
        
        return SymmetryResult(
            name=f"Glide ({axis_type})",
            present=present,
            score=score,
            threshold=self.threshold,
            details=f"Glide reflection along {axis_type} axis"
        )
    
    def _check_unexpected_symmetries(self, 
                                      cell: np.ndarray, 
                                      expected_group: str) -> List[str]:
        """Check for symmetries that would indicate a higher symmetry group."""
        unexpected = []
        props = GROUP_PROPERTIES[expected_group]
        
        # If rotation order is n, check for higher orders
        if props['rotation'] < 4:
            rot4 = self._check_rotation_symmetry(cell, 4)
            if rot4.present:
                unexpected.append("4-fold rotation (suggests p4 group)")
        
        if props['rotation'] < 6:
            rot6 = self._check_rotation_symmetry(cell, 6)
            if rot6.present:
                unexpected.append("6-fold rotation (suggests p6 group)")
        
        # If no reflection expected, check for it
        if not props['reflections']:
            ref_h = self._check_reflection_symmetry(cell, 'h')
            ref_v = self._check_reflection_symmetry(cell, 'v')
            if ref_h.present or ref_v.present:
                unexpected.append("reflection (suggests higher group)")
        
        return unexpected
    
    def verify_batch(self,
                     images: List[np.ndarray],
                     expected_group: str,
                     known_period: Optional[Tuple[int, int]] = None,
                     verbose: bool = False) -> Dict:
        """Verify a batch of images."""
        results = []
        
        for i, img in enumerate(images):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Verifying {i+1}/{len(images)}...")
            
            result = self.verify(img, expected_group, known_period, verbose=False)
            results.append(result)
        
        n_verified = sum(1 for r in results if r.verified)
        n_total = len(results)
        
        # Symmetry stats
        symmetry_stats = {}
        if results and results[0].symmetry_scores:
            for sym_name in results[0].symmetry_scores.keys():
                scores = [r.symmetry_scores[sym_name].score for r in results 
                         if sym_name in r.symmetry_scores]
                if scores:
                    symmetry_stats[sym_name] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'pass_rate': float(np.mean([s >= self.threshold for s in scores]))
                    }
        
        return {
            'expected_group': expected_group,
            'n_verified': n_verified,
            'n_total': n_total,
            'pass_rate': n_verified / n_total if n_total > 0 else 0,
            'avg_confidence': float(np.mean([r.confidence for r in results])),
            'symmetry_stats': symmetry_stats,
            'results': results
        }


def verify_dataset(data_path: str,
                   output_path: Optional[str] = None,
                   max_samples_per_group: int = 50,
                   known_period: Optional[int] = None,
                   verbose: bool = True) -> Dict[str, Dict]:
    """
    Verify symmetries for a crystallographic dataset.
    
    Args:
        data_path: Path to HDF5 or image directory
        output_path: Path to save JSON results
        max_samples_per_group: Max samples to check per group
        known_period: If known, the motif size used in generation
        verbose: Print progress
    """
    import h5py
    import json
    
    verifier = SymmetryVerifier(similarity_threshold=0.70)
    results = {}
    
    data_path = Path(data_path)
    
    if data_path.suffix == '.h5':
        with h5py.File(data_path, 'r') as f:
            patterns = f['patterns'][:]
            labels = f['labels'][:]
            group_names = [g.decode('utf-8') for g in f['group_names'][:]]
            
            # Try to get motif size from metadata
            if known_period is None and 'metadata' in f and 'motif_size' in f['metadata']:
                motif_sizes = f['metadata']['motif_size'][:]
                known_period = int(np.median(motif_sizes))
                if verbose:
                    print(f"Detected motif size from metadata: {known_period}")
            
            for group_idx, group_name in enumerate(group_names):
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Verifying: {group_name}")
                    print('='*50)
                
                mask = labels == group_idx
                group_patterns = patterns[mask][:max_samples_per_group]
                
                # Use known period if available
                period = (known_period, known_period) if known_period else None
                
                batch_result = verifier.verify_batch(
                    [p for p in group_patterns],
                    group_name,
                    known_period=period,
                    verbose=verbose
                )
                
                results[group_name] = batch_result
                
                if verbose:
                    print(f"\n  Results for {group_name}:")
                    print(f"    Pass rate: {batch_result['pass_rate']*100:.1f}%")
                    print(f"    Avg confidence: {batch_result['avg_confidence']:.3f}")
                    for sym, stats in batch_result['symmetry_stats'].items():
                        print(f"    {sym}: mean={stats['mean']:.3f}, pass={stats['pass_rate']*100:.0f}%")
    
    elif data_path.is_dir():
        images_dir = data_path / 'images' if (data_path / 'images').exists() else data_path
        
        for group_dir in sorted(images_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            
            group_name = group_dir.name
            if group_name not in GROUP_PROPERTIES:
                continue
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"Verifying: {group_name}")
                print('='*50)
            
            from PIL import Image
            images = []
            for img_path in sorted(group_dir.glob('*.png'))[:max_samples_per_group]:
                img = np.array(Image.open(img_path)) / 255.0
                images.append(img)
            
            if not images:
                continue
            
            period = (known_period, known_period) if known_period else None
            batch_result = verifier.verify_batch(images, group_name, period, verbose)
            results[group_name] = batch_result
            
            if verbose:
                print(f"\n  Pass rate: {batch_result['pass_rate']*100:.1f}%")
    
    # Save results
    if output_path:
        serializable = {}
        for group, data in results.items():
            serializable[group] = {
                'expected_group': data['expected_group'],
                'n_verified': data['n_verified'],
                'n_total': data['n_total'],
                'pass_rate': float(data['pass_rate']),
                'avg_confidence': float(data['avg_confidence']),
                'symmetry_stats': data['symmetry_stats']
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return results


def create_verification_report(results: Dict[str, Dict], output_path: str):
    """Create visual report."""
    import matplotlib.pyplot as plt
    
    groups = list(results.keys())
    n_groups = len(groups)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Symmetry Verification Report', fontsize=14, fontweight='bold')
    
    # Pass rate
    ax1 = axes[0]
    pass_rates = [results[g]['pass_rate'] * 100 for g in groups]
    colors = ['#2ecc71' if r >= 70 else '#f39c12' if r >= 40 else '#e74c3c' for r in pass_rates]
    
    ax1.bar(range(len(groups)), pass_rates, color=colors)
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, rotation=45, ha='right')
    ax1.set_ylabel('Pass Rate (%)')
    ax1.set_title('Verification Pass Rate')
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 105)
    
    # Confidence
    ax2 = axes[1]
    confidences = [results[g]['avg_confidence'] for g in groups]
    ax2.bar(range(len(groups)), confidences, color='#3498db')
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels(groups, rotation=45, ha='right')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Symmetry Confidence')
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify crystallographic symmetries")
    parser.add_argument('data_path', type=str, help="Path to dataset")
    parser.add_argument('--output', '-o', type=str, default='verification_results.json')
    parser.add_argument('--report', '-r', type=str, default='verification_report.png')
    parser.add_argument('--max-samples', '-n', type=int, default=50)
    parser.add_argument('--period', '-p', type=int, default=None,
                       help="Known motif size/period (if known from generation)")
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    
    results = verify_dataset(
        args.data_path,
        output_path=args.output,
        max_samples_per_group=args.max_samples,
        known_period=args.period,
        verbose=not args.quiet
    )
    
    create_verification_report(results, args.report)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for group, data in sorted(results.items()):
        status = "✓" if data['pass_rate'] >= 0.7 else "⚠" if data['pass_rate'] >= 0.4 else "✗"
        print(f"  {status} {group:6s}: {data['pass_rate']*100:5.1f}% (conf: {data['avg_confidence']:.3f})")

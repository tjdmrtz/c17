#!/usr/bin/env python3
"""
Script to visualize crystallographic patterns.

Generates beautiful visualizations of the 17 wallpaper groups.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt

from src.dataset.pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS
from src.visualization.visualize import (
    PatternVisualizer,
    plot_all_groups,
    plot_group_comparison,
    plot_lattice_types
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize crystallographic patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output/visualizations",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=256,
        help="Pattern resolution"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--style",
        choices=['dark', 'light'],
        default='dark',
        help="Visualization style"
    )
    
    parser.add_argument(
        "--groups",
        nargs='+',
        default=None,
        help="Specific groups to visualize (default: all)"
    )
    
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Generate the complete 17-group overview"
    )
    
    parser.add_argument(
        "--by-lattice",
        action="store_true",
        help="Generate visualization organized by lattice type"
    )
    
    parser.add_argument(
        "--with-symmetry",
        action="store_true",
        help="Generate patterns with symmetry annotations"
    )
    
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate comparison of multiple samples"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Style: {args.style}")
    print("-" * 60)
    
    # If no specific visualizations requested, do all
    do_all = not any([args.all_groups, args.by_lattice, args.with_symmetry, 
                      args.comparison, args.groups])
    
    # Generate all 17 groups overview
    if args.all_groups or do_all:
        print("\nGenerating all 17 wallpaper groups overview...")
        fig = plot_all_groups(
            resolution=args.resolution,
            seed=args.seed,
            save_path=str(output_path / "all_17_groups.png"),
            style=args.style
        )
        plt.close(fig)
        print("  ✓ Saved: all_17_groups.png")
    
    # Organize by lattice type
    if args.by_lattice or do_all:
        print("\nGenerating lattice type visualization...")
        fig = plot_lattice_types(
            resolution=args.resolution,
            seed=args.seed,
            save_path=str(output_path / "by_lattice_type.png")
        )
        plt.close(fig)
        print("  ✓ Saved: by_lattice_type.png")
    
    # Comparison view
    if args.comparison or do_all:
        print("\nGenerating pattern comparisons...")
        groups_to_compare = args.groups if args.groups else ['p1', 'p4', 'p6m']
        fig = plot_group_comparison(
            groups_to_compare,
            num_samples=5,
            resolution=args.resolution,
            seed=args.seed,
            save_path=str(output_path / "group_comparison.png")
        )
        plt.close(fig)
        print("  ✓ Saved: group_comparison.png")
    
    # Individual patterns with symmetry annotations
    if args.with_symmetry or do_all:
        print("\nGenerating symmetry annotations...")
        visualizer = PatternVisualizer(style=args.style)
        generator = WallpaperGroupGenerator(resolution=args.resolution, seed=args.seed)
        
        groups_to_annotate = args.groups if args.groups else ['p2', 'pmm', 'p4m', 'p6m']
        
        for group_name in groups_to_annotate:
            if group_name not in WALLPAPER_GROUPS:
                print(f"  ⚠ Unknown group: {group_name}, skipping")
                continue
                
            pattern = generator.generate(group_name, motif_size=args.resolution // 4)
            
            # Pattern with info
            fig = visualizer.plot_single_pattern(
                pattern, group_name,
                save_path=str(output_path / f"pattern_{group_name}.png")
            )
            plt.close(fig)
            
            # Symmetry annotations
            fig = visualizer.plot_symmetry_annotations(
                pattern, group_name,
                save_path=str(output_path / f"symmetry_{group_name}.png")
            )
            plt.close(fig)
            
            print(f"  ✓ Saved: pattern_{group_name}.png, symmetry_{group_name}.png")
    
    # Individual groups if specified
    if args.groups and not (args.with_symmetry or args.comparison):
        print(f"\nGenerating individual patterns for: {', '.join(args.groups)}")
        visualizer = PatternVisualizer(style=args.style)
        generator = WallpaperGroupGenerator(resolution=args.resolution, seed=args.seed)
        
        for group_name in args.groups:
            if group_name not in WALLPAPER_GROUPS:
                print(f"  ⚠ Unknown group: {group_name}, skipping")
                continue
            
            pattern = generator.generate(group_name, motif_size=args.resolution // 4)
            fig = visualizer.plot_single_pattern(
                pattern, group_name,
                save_path=str(output_path / f"pattern_{group_name}.png")
            )
            plt.close(fig)
            print(f"  ✓ Saved: pattern_{group_name}.png")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"All visualizations saved to: {output_path}")


if __name__ == "__main__":
    main()









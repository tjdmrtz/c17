"""
PyTorch Dataset for Colored Crystallographic Patterns.

Supports RGB color patterns with 3 channels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import h5py

from .color_patterns import ColorPatternGenerator, GROUP_PALETTES
from .pattern_generator import WALLPAPER_GROUPS


class ColoredCrystallographicDataset(Dataset):
    """
    Dataset of colored crystallographic patterns for the 17 wallpaper groups.
    
    Each group has a unique color palette, making patterns visually distinctive.
    Output format: [3, H, W] RGB tensors with values in [0, 1].
    """
    
    GROUP_TO_IDX = {name: idx for idx, name in enumerate(WALLPAPER_GROUPS.keys())}
    IDX_TO_GROUP = {idx: name for name, idx in GROUP_TO_IDX.items()}
    
    def __init__(self,
                 num_samples_per_group: int = 100,
                 resolution: int = 128,
                 motif_size: int = 32,
                 transform: Optional[Any] = None,
                 seed: Optional[int] = None,
                 pregenerate: bool = True,
                 cache_path: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            num_samples_per_group: Number of samples to generate per group
            resolution: Resolution of output images
            motif_size: Size of fundamental motif
            transform: Optional transform to apply
            seed: Random seed for reproducibility
            pregenerate: Whether to generate all samples upfront
            cache_path: Path to cache/load the dataset (HDF5 format)
        """
        self.num_samples_per_group = num_samples_per_group
        self.resolution = resolution
        self.motif_size = motif_size
        self.transform = transform
        self.seed = seed
        self.num_groups = len(WALLPAPER_GROUPS)
        
        self.samples: List[Tuple[np.ndarray, int]] = []
        
        if cache_path and Path(cache_path).exists():
            self._load_from_cache(cache_path)
        elif pregenerate:
            self._generate_all_samples()
            if cache_path:
                self._save_to_cache(cache_path)
    
    def _generate_all_samples(self):
        """Generate all colored samples for the dataset."""
        from tqdm import tqdm
        
        print(f"Generating {self.num_samples_per_group * self.num_groups} colored samples...")
        
        for group_idx, group_name in enumerate(WALLPAPER_GROUPS.keys()):
            palette = GROUP_PALETTES[group_name]
            
            for sample_idx in tqdm(range(self.num_samples_per_group), 
                                   desc=f"Generating {group_name} ({palette['name']})"):
                # Use deterministic seed for each sample
                sample_seed = None
                if self.seed is not None:
                    sample_seed = self.seed + group_idx * 10000 + sample_idx
                
                generator = ColorPatternGenerator(
                    resolution=self.resolution,
                    seed=sample_seed
                )
                
                # Vary motif type and complexity
                motif_types = ["gaussian", "geometric", "mixed"]
                motif_type = motif_types[sample_idx % len(motif_types)]
                complexity = (sample_idx % 5) + 2
                
                pattern = generator.generate(
                    group_name,
                    motif_size=self.motif_size,
                    complexity=complexity,
                    motif_type=motif_type
                )
                
                self.samples.append((pattern, group_idx))
    
    def _save_to_cache(self, cache_path: str):
        """Save dataset to HDF5 cache."""
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(cache_path, 'w') as f:
            patterns = np.array([s[0] for s in self.samples])
            labels = np.array([s[1] for s in self.samples])
            
            f.create_dataset('patterns', data=patterns, compression='gzip')
            f.create_dataset('labels', data=labels)
            f.attrs['num_samples_per_group'] = self.num_samples_per_group
            f.attrs['resolution'] = self.resolution
            f.attrs['motif_size'] = self.motif_size
            f.attrs['channels'] = 3
            
        print(f"Dataset cached to {cache_path}")
    
    def _load_from_cache(self, cache_path: str):
        """Load dataset from HDF5 cache."""
        print(f"Loading dataset from {cache_path}...")
        
        with h5py.File(cache_path, 'r') as f:
            patterns = f['patterns'][:]
            labels = f['labels'][:]
            
            self.samples = [(patterns[i], labels[i]) for i in range(len(labels))]
            
        print(f"Loaded {len(self.samples)} colored samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pattern, label = self.samples[idx]
        
        # Convert to tensor with channel-first format [3, H, W]
        tensor = torch.from_numpy(pattern).float().permute(2, 0, 1)
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label
    
    def get_group_name(self, idx: int) -> str:
        """Get the wallpaper group name for a given label index."""
        return self.IDX_TO_GROUP[idx]
    
    def get_palette_name(self, idx: int) -> str:
        """Get the palette name for a given label index."""
        group_name = self.IDX_TO_GROUP[idx]
        return GROUP_PALETTES[group_name]['name']


def create_colored_dataloaders(
    batch_size: int = 32,
    num_samples_per_group: int = 100,
    resolution: int = 128,
    train_split: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for colored patterns.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = ColoredCrystallographicDataset(
        num_samples_per_group=num_samples_per_group,
        resolution=resolution,
        seed=seed,
        **kwargs
    )
    
    # Split dataset
    total_samples = len(dataset)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader







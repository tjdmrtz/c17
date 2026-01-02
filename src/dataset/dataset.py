"""
PyTorch Dataset for Crystallographic Patterns.

Provides both image-based and graph-based representations for use
with CNNs and GCNs respectively.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import h5py

from .pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS


class CrystallographicDataset(Dataset):
    """
    Dataset of crystallographic patterns for the 17 wallpaper groups.
    
    Supports two representations:
    - Image (for CNN): 2D grayscale images
    - Graph (for GCN): Nodes with features and adjacency
    """
    
    GROUP_TO_IDX = {name: idx for idx, name in enumerate(WALLPAPER_GROUPS.keys())}
    IDX_TO_GROUP = {idx: name for name, idx in GROUP_TO_IDX.items()}
    
    def __init__(self,
                 num_samples_per_group: int = 100,
                 resolution: int = 128,
                 motif_size: int = 32,
                 representation: str = "image",
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
            representation: "image" for CNN or "graph" for GCN
            transform: Optional transform to apply
            seed: Random seed for reproducibility
            pregenerate: Whether to generate all samples upfront
            cache_path: Path to cache/load the dataset (HDF5 format)
        """
        self.num_samples_per_group = num_samples_per_group
        self.resolution = resolution
        self.motif_size = motif_size
        self.representation = representation
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
        """Generate all samples for the dataset."""
        from tqdm import tqdm
        
        print(f"Generating {self.num_samples_per_group * self.num_groups} samples...")
        
        for group_idx, group_name in enumerate(WALLPAPER_GROUPS.keys()):
            for sample_idx in tqdm(range(self.num_samples_per_group), 
                                   desc=f"Generating {group_name}"):
                # Use deterministic seed for each sample
                sample_seed = None
                if self.seed is not None:
                    sample_seed = self.seed + group_idx * 10000 + sample_idx
                
                generator = WallpaperGroupGenerator(
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
            
        print(f"Dataset cached to {cache_path}")
    
    def _load_from_cache(self, cache_path: str):
        """Load dataset from HDF5 cache."""
        print(f"Loading dataset from {cache_path}...")
        
        with h5py.File(cache_path, 'r') as f:
            patterns = f['patterns'][:]
            labels = f['labels'][:]
            
            self.samples = [(patterns[i], labels[i]) for i in range(len(labels))]
            
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pattern, label = self.samples[idx]
        
        if self.representation == "image":
            # Convert to tensor with channel dimension [1, H, W]
            tensor = torch.from_numpy(pattern).float().unsqueeze(0)
        elif self.representation == "graph":
            tensor = self._pattern_to_graph(pattern)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label
    
    def _pattern_to_graph(self, pattern: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Convert pattern to graph representation.
        
        The graph is constructed by:
        1. Sampling points from the pattern
        2. Creating edges based on spatial proximity
        3. Using pixel values as node features
        """
        # Downsample for graph representation
        stride = max(1, self.resolution // 32)
        points = []
        features = []
        
        for i in range(0, pattern.shape[0], stride):
            for j in range(0, pattern.shape[1], stride):
                points.append([i / pattern.shape[0], j / pattern.shape[1]])
                
                # Node features: intensity + local gradient
                val = pattern[i, j]
                grad_x = pattern[min(i+1, pattern.shape[0]-1), j] - pattern[max(i-1, 0), j]
                grad_y = pattern[i, min(j+1, pattern.shape[1]-1)] - pattern[i, max(j-1, 0)]
                features.append([val, grad_x, grad_y])
        
        points = np.array(points)
        features = np.array(features)
        
        # Build edges (k-nearest neighbors + grid adjacency)
        from scipy.spatial import distance_matrix
        dist_mat = distance_matrix(points, points)
        
        # Connect nodes within a threshold distance
        threshold = 2.0 * stride / pattern.shape[0]
        edge_index = []
        for i in range(len(points)):
            neighbors = np.where((dist_mat[i] < threshold) & (dist_mat[i] > 0))[0]
            for j in neighbors:
                edge_index.append([i, j])
        
        edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        
        return {
            'x': torch.from_numpy(features).float(),
            'edge_index': torch.from_numpy(edge_index).long(),
            'pos': torch.from_numpy(points).float(),
        }
    
    def get_group_name(self, idx: int) -> str:
        """Get the wallpaper group name for a given label index."""
        return self.IDX_TO_GROUP[idx]
    
    def get_samples_by_group(self, group_name: str) -> List[np.ndarray]:
        """Get all samples for a specific wallpaper group."""
        group_idx = self.GROUP_TO_IDX[group_name]
        return [pattern for pattern, label in self.samples if label == group_idx]


def create_dataloaders(
    batch_size: int = 32,
    num_samples_per_group: int = 100,
    resolution: int = 128,
    train_split: float = 0.8,
    seed: int = 42,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = CrystallographicDataset(
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader









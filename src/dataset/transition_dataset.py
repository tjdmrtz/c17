"""
Transition Dataset for Neural ODE Training.

Creates pairs of (source_pattern, target_pattern) for learning
phase transitions between wallpaper groups.

Uses the real crystallographic patterns from H5 dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm


# All 17 wallpaper groups
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]

GROUP_TO_IDX = {g: i for i, g in enumerate(ALL_17_GROUPS)}
IDX_TO_GROUP = {i: g for i, g in enumerate(ALL_17_GROUPS)}


class H5PatternDataset(Dataset):
    """
    Dataset that loads crystallographic patterns from H5 file.
    
    Returns RGB images normalized to [0, 1].
    Preloads everything into RAM for maximum speed.
    """
    
    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        splits_path: Optional[str] = None,
        transform: Optional[callable] = None,
        use_cleaned: bool = True,
        preload: bool = True,
    ):
        """
        Args:
            h5_path: Path to H5 file
            split: 'train', 'val', or 'test'
            splits_path: Path to splits.npz (default: same dir as h5)
            transform: Optional transform to apply
            use_cleaned: If True, use cleaned splits (verified symmetry)
            preload: If True, load entire dataset into RAM (faster training)
        """
        self.h5_path = Path(h5_path)
        self.split = split
        self.transform = transform
        self.preload = preload
        
        # Load splits (prefer cleaned if available)
        if splits_path is None:
            cleaned_path = self.h5_path.parent / 'splits_cleaned.npz'
            original_path = self.h5_path.parent / 'splits.npz'
            
            if use_cleaned and cleaned_path.exists():
                splits_path = cleaned_path
                print(f"Using cleaned splits: {cleaned_path}")
            else:
                splits_path = original_path
        
        splits = np.load(splits_path)
        self.indices = splits[split]
        
        # Open H5 file and load data
        with h5py.File(self.h5_path, 'r') as f:
            self.labels = f['labels'][:]
            self.group_names = [g.decode() for g in f['group_names'][:]]
            
            if preload:
                # Preload all patterns for this split into RAM
                print(f"Preloading {len(self.indices)} patterns into RAM...")
                all_patterns = f['patterns'][:]
                # Normalize and convert to (C, H, W) format
                if all_patterns.max() > 1.0:
                    all_patterns = all_patterns.astype(np.float32) / 255.0
                else:
                    all_patterns = all_patterns.astype(np.float32)
                # Transpose from (N, H, W, C) to (N, C, H, W)
                all_patterns = np.transpose(all_patterns, (0, 3, 1, 2))
                # Convert to tensor and keep only our split indices
                self._patterns_cache = torch.from_numpy(all_patterns)
                print(f"Preloaded {len(self.indices)} patterns ({self._patterns_cache.nbytes / 1e9:.2f} GB)")
            else:
                self._patterns_cache = None
                self.h5_file = h5py.File(self.h5_path, 'r')
                self.patterns = self.h5_file['patterns']
        
        # Get indices per group for efficient sampling
        self.indices_by_group = {}
        for group_idx in range(17):
            mask = self.labels[self.indices] == group_idx
            self.indices_by_group[group_idx] = self.indices[mask]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            pattern: (3, 256, 256) tensor
            label: group index
            group_name: group name string
        """
        real_idx = self.indices[idx]
        
        if self.preload:
            pattern = self._patterns_cache[real_idx].clone()
        else:
            # Load from disk
            pattern = self.patterns[real_idx]
            if pattern.max() > 1.0:
                pattern = pattern / 255.0
            pattern = np.transpose(pattern, (2, 0, 1)).astype(np.float32)
            pattern = torch.from_numpy(pattern)
        
        label = int(self.labels[real_idx])
        group_name = self.group_names[label]
        
        if self.transform:
            pattern = self.transform(pattern)
        
        return pattern, label, group_name
    
    def get_pattern_by_group(self, group_idx: int, sample_idx: int = 0):
        """Get a specific pattern from a group."""
        indices = self.indices_by_group[group_idx]
        if len(indices) == 0:
            raise ValueError(f"No patterns for group {group_idx}")
        
        real_idx = indices[sample_idx % len(indices)]
        
        if self.preload:
            return self._patterns_cache[real_idx].clone()
        else:
            pattern = self.patterns[real_idx]
            if pattern.max() > 1.0:
                pattern = pattern / 255.0
            pattern = np.transpose(pattern, (2, 0, 1)).astype(np.float32)
            return torch.from_numpy(pattern)
    
    def get_random_pattern_by_group(self, group_idx: int, rng: np.random.Generator = None):
        """Get a random pattern from a group."""
        if rng is None:
            rng = np.random.default_rng()
        
        indices = self.indices_by_group[group_idx]
        if len(indices) == 0:
            raise ValueError(f"No patterns for group {group_idx}")
        
        sample_idx = rng.integers(0, len(indices))
        return self.get_pattern_by_group(group_idx, sample_idx)
    
    def close(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
    
    def __del__(self):
        self.close()


class TransitionDataset(Dataset):
    """
    Dataset of transition pairs for Neural ODE training.
    
    Uses precomputed latent representations from VAE.
    """
    
    def __init__(
        self,
        vae,
        h5_dataset: H5PatternDataset,
        device: torch.device,
        pairs_per_epoch: int = 10000,
        transition_mode: str = 'cross',  # 'cross', 'same_z', 'mixed'
        precompute_latents: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            vae: Pretrained VAE model
            h5_dataset: H5PatternDataset instance
            device: Device for computation
            pairs_per_epoch: Number of pairs to generate
            transition_mode: 'cross' = different patterns, 
                           'same_z' = same base latent decoded differently
                           'mixed' = 50/50
            precompute_latents: Whether to precompute all latents
            seed: Random seed
        """
        self.vae = vae
        self.h5_dataset = h5_dataset
        self.device = device
        self.pairs_per_epoch = pairs_per_epoch
        self.transition_mode = transition_mode
        self.rng = np.random.default_rng(seed)
        
        # Storage for latent representations
        self.latents = {}  # {group_idx: [(pattern_idx, z), ...]}
        self.patterns_cache = {}  # {(group_idx, pattern_idx): pattern}
        
        if precompute_latents:
            self._precompute_latents()
        
        # Generate transition pairs
        self.pairs = []
        self._generate_pairs()
    
    def _precompute_latents(self):
        """Precompute latent representations for all patterns."""
        print("Precomputing latent representations...")
        
        self.vae.eval()
        
        for group_idx in range(17):
            group_name = IDX_TO_GROUP[group_idx]
            self.latents[group_idx] = []
            
            indices = self.h5_dataset.indices_by_group[group_idx]
            
            if len(indices) == 0:
                continue
            
            # Process in batches
            batch_size = 32
            for start in tqdm(range(0, len(indices), batch_size), 
                             desc=f"Group {group_name}", leave=False):
                end = min(start + batch_size, len(indices))
                batch_indices = list(range(start, end))
                
                patterns = []
                for i in batch_indices:
                    pattern = self.h5_dataset.get_pattern_by_group(group_idx, i)
                    patterns.append(pattern)
                
                patterns = torch.stack(patterns).to(self.device)
                
                with torch.no_grad():
                    z = self.vae.encode(patterns, group_name)
                
                for i, idx in enumerate(batch_indices):
                    self.latents[group_idx].append((idx, z[i].cpu()))
        
        print(f"Precomputed {sum(len(v) for v in self.latents.values())} latents")
    
    def _generate_pairs(self):
        """Generate transition pairs."""
        self.pairs = []
        
        for _ in range(self.pairs_per_epoch):
            # Random source and target groups
            source_idx = self.rng.integers(0, 17)
            target_idx = self.rng.integers(0, 17)
            
            # Get random samples from each group
            if len(self.latents.get(source_idx, [])) == 0:
                continue
            if len(self.latents.get(target_idx, [])) == 0:
                continue
            
            source_sample = self.rng.integers(0, len(self.latents[source_idx]))
            target_sample = self.rng.integers(0, len(self.latents[target_idx]))
            
            source_pattern_idx, z_source = self.latents[source_idx][source_sample]
            target_pattern_idx, z_target = self.latents[target_idx][target_sample]
            
            self.pairs.append({
                'z_source': z_source,
                'z_target': z_target,
                'source_idx': source_idx,
                'target_idx': target_idx,
                'source_pattern_idx': source_pattern_idx,
                'target_pattern_idx': target_pattern_idx,
            })
    
    def regenerate_pairs(self):
        """Regenerate pairs (call each epoch for variety)."""
        self._generate_pairs()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return (
            pair['z_source'],
            pair['z_target'],
            pair['source_idx'],
            pair['target_idx'],
        )


class TransitionDatasetBatched(Dataset):
    """
    Batched transition dataset that groups by source/target pairs.
    
    This ensures all samples in a batch have the same source/target groups,
    avoiding the bug in the original implementation.
    """
    
    def __init__(
        self,
        vae,
        h5_dataset: H5PatternDataset,
        device: torch.device,
        samples_per_pair: int = 100,
        precompute_latents: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            vae: Pretrained VAE
            h5_dataset: H5PatternDataset
            device: Device
            samples_per_pair: Samples per group pair (17*17=289 pairs)
            precompute_latents: Precompute latents
            seed: Random seed
        """
        self.vae = vae
        self.h5_dataset = h5_dataset
        self.device = device
        self.samples_per_pair = samples_per_pair
        self.rng = np.random.default_rng(seed)
        
        self.latents = {}
        if precompute_latents:
            self._precompute_latents()
        
        # Create all pairs organized by (source, target)
        self.pair_data = {}  # {(source_idx, target_idx): [(z_s, z_t), ...]}
        self._organize_pairs()
    
    def _precompute_latents(self):
        """Precompute latents."""
        print("Precomputing latent representations...")
        self.vae.eval()
        
        for group_idx in range(17):
            group_name = IDX_TO_GROUP[group_idx]
            self.latents[group_idx] = []
            
            indices = self.h5_dataset.indices_by_group[group_idx]
            if len(indices) == 0:
                continue
            
            batch_size = 32
            for start in tqdm(range(0, len(indices), batch_size),
                             desc=f"Group {group_name}", leave=False):
                end = min(start + batch_size, len(indices))
                
                patterns = []
                for i in range(start, end):
                    pattern = self.h5_dataset.get_pattern_by_group(group_idx, i)
                    patterns.append(pattern)
                
                patterns = torch.stack(patterns).to(self.device)
                
                with torch.no_grad():
                    z = self.vae.encode(patterns, group_name)
                
                for i in range(z.shape[0]):
                    self.latents[group_idx].append(z[i].cpu())
        
        print(f"Precomputed latents for {sum(len(v) for v in self.latents.values())} patterns")
    
    def _organize_pairs(self):
        """Organize pairs by (source, target) groups."""
        for source_idx in range(17):
            for target_idx in range(17):
                key = (source_idx, target_idx)
                self.pair_data[key] = []
                
                if len(self.latents.get(source_idx, [])) == 0:
                    continue
                if len(self.latents.get(target_idx, [])) == 0:
                    continue
                
                # Generate samples for this pair
                for _ in range(self.samples_per_pair):
                    s_idx = self.rng.integers(0, len(self.latents[source_idx]))
                    t_idx = self.rng.integers(0, len(self.latents[target_idx]))
                    
                    z_source = self.latents[source_idx][s_idx]
                    z_target = self.latents[target_idx][t_idx]
                    
                    self.pair_data[key].append((z_source, z_target))
        
        # Create flat list for iteration
        self.samples = []
        for (source_idx, target_idx), pairs in self.pair_data.items():
            for z_source, z_target in pairs:
                self.samples.append({
                    'z_source': z_source,
                    'z_target': z_target,
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample['z_source'],
            sample['z_target'],
            sample['source_idx'],
            sample['target_idx'],
        )
    
    def get_batch_for_pair(self, source_idx: int, target_idx: int, 
                           batch_size: int = 32) -> Tuple[torch.Tensor, ...]:
        """Get a batch of samples for a specific group pair."""
        pairs = self.pair_data.get((source_idx, target_idx), [])
        
        if len(pairs) == 0:
            return None
        
        indices = self.rng.choice(len(pairs), size=min(batch_size, len(pairs)), replace=False)
        
        z_sources = torch.stack([pairs[i][0] for i in indices])
        z_targets = torch.stack([pairs[i][1] for i in indices])
        source_indices = torch.full((len(indices),), source_idx, dtype=torch.long)
        target_indices = torch.full((len(indices),), target_idx, dtype=torch.long)
        
        return z_sources, z_targets, source_indices, target_indices


def create_transition_dataloader(
    vae,
    h5_path: str,
    split: str = 'train',
    batch_size: int = 64,
    pairs_per_epoch: int = 10000,
    device: torch.device = None,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for transition training.
    
    Args:
        vae: Pretrained VAE
        h5_path: Path to H5 dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        pairs_per_epoch: Pairs per epoch
        device: Device for VAE
        seed: Random seed
        num_workers: DataLoader workers
        
    Returns:
        DataLoader for transition pairs
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    h5_dataset = H5PatternDataset(h5_path, split=split)
    
    transition_dataset = TransitionDataset(
        vae=vae,
        h5_dataset=h5_dataset,
        device=device,
        pairs_per_epoch=pairs_per_epoch,
        seed=seed,
    )
    
    dataloader = DataLoader(
        transition_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, transition_dataset


if __name__ == "__main__":
    # Test dataset loading
    print("Testing H5PatternDataset...")
    
    h5_path = "data/colored_crystallographic/crystallographic_patterns_colored.h5"
    
    dataset = H5PatternDataset(h5_path, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    pattern, label, group_name = dataset[0]
    print(f"Pattern shape: {pattern.shape}")
    print(f"Label: {label}, Group: {group_name}")
    print(f"Value range: [{pattern.min():.3f}, {pattern.max():.3f}]")
    
    dataset.close()


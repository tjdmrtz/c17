from .vae import CrystallographicVAE, VAELoss
from .vae_medium import CrystallographicVAEMedium, get_model_for_dataset_size
from .vae_large import CrystallographicVAELarge
from .vae_deep import DeepCrystallographicVAE
from .vae_simple_rgb import SimpleVAE, SimpleVAEConfig, SimpleVAELoss
from .flow_matching_transition import (
    FlowMatchingTransition, 
    FlowMatchingConfig,
    FlowMatchingMetrics,
    ALL_17_GROUPS,
    GROUP_TO_IDX,
    IDX_TO_GROUP,
)
from .trainer import VAETrainer

__all__ = [
    # Legacy VAE models
    'CrystallographicVAE', 
    'CrystallographicVAEMedium',
    'CrystallographicVAELarge',
    'DeepCrystallographicVAE',
    'VAELoss', 
    'VAETrainer',
    'get_model_for_dataset_size',
    # Simple RGB VAE (recommended)
    'SimpleVAE',
    'SimpleVAEConfig',
    'SimpleVAELoss',
    # Flow Matching (state-of-the-art)
    'FlowMatchingTransition',
    'FlowMatchingConfig',
    'FlowMatchingMetrics',
    'ALL_17_GROUPS',
    'GROUP_TO_IDX',
    'IDX_TO_GROUP',
]


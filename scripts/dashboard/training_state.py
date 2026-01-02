"""
Shared state between training loop and dashboard.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import threading


@dataclass
class TrainingState:
    """
    Shared state for training monitoring.
    
    This object is updated by the training loop and read by the dashboard.
    Thread-safe with a lock for concurrent access.
    """
    
    # Training progress
    epoch: int = 0
    total_epochs: int = 100
    batch: int = 0
    total_batches: int = 0
    
    # Current metrics
    current_loss: float = 0.0
    current_lr: float = 0.0
    
    # History
    losses: Dict[str, List[float]] = field(default_factory=lambda: {
        'total': [],
        'endpoint': [],
        'smoothness': [],
        'velocity': [],
    })
    
    # Timestamps
    epoch_times: List[float] = field(default_factory=list)
    start_time: Optional[float] = None
    
    # Visualization paths
    output_dir: Optional[Path] = None
    latest_latent_viz: Optional[str] = None
    latest_recon_viz: Optional[str] = None
    latest_transitions: Dict[str, str] = field(default_factory=dict)
    
    # Log messages
    log_messages: List[str] = field(default_factory=list)
    max_log_messages: int = 500
    
    # Status
    is_training: bool = False
    is_paused: bool = False
    error_message: Optional[str] = None
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def update_loss(self, epoch: int, losses: Dict[str, float]):
        """Update loss history."""
        with self._lock:
            self.epoch = epoch
            self.current_loss = losses.get('loss', 0.0)
            
            for key in ['total', 'endpoint', 'smoothness', 'velocity']:
                loss_key = key if key == 'total' else f'{key}_loss'
                if loss_key in losses or key in losses:
                    value = losses.get(loss_key, losses.get(key, 0.0))
                    if isinstance(value, (int, float)):
                        self.losses[key].append(float(value))
    
    def update_batch(self, batch: int, total_batches: int, loss: float):
        """Update batch progress."""
        with self._lock:
            self.batch = batch
            self.total_batches = total_batches
            self.current_loss = loss
    
    def log(self, message: str):
        """Add log message."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] {message}"
        
        with self._lock:
            self.log_messages.append(formatted)
            if len(self.log_messages) > self.max_log_messages:
                self.log_messages = self.log_messages[-self.max_log_messages:]
    
    def set_visualization(self, viz_type: str, path: str):
        """Set visualization path."""
        with self._lock:
            if viz_type == 'latent':
                self.latest_latent_viz = path
            elif viz_type == 'recon':
                self.latest_recon_viz = path
            elif viz_type.startswith('transition_'):
                key = viz_type.replace('transition_', '')
                self.latest_transitions[key] = path
    
    def get_progress(self) -> float:
        """Get overall training progress [0, 1]."""
        with self._lock:
            if self.total_epochs == 0:
                return 0.0
            epoch_progress = self.epoch / self.total_epochs
            if self.total_batches > 0:
                batch_progress = self.batch / self.total_batches / self.total_epochs
                return epoch_progress + batch_progress
            return epoch_progress
    
    def get_eta_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        with self._lock:
            if len(self.epoch_times) < 2:
                return None
            
            avg_epoch_time = sum(self.epoch_times[-10:]) / len(self.epoch_times[-10:])
            remaining_epochs = self.total_epochs - self.epoch
            return avg_epoch_time * remaining_epochs
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        with self._lock:
            return {
                'epoch': self.epoch,
                'total_epochs': self.total_epochs,
                'current_loss': self.current_loss,
                'current_lr': self.current_lr,
                'losses': self.losses,
                'is_training': self.is_training,
            }
    
    def save(self, path: Path):
        """Save state to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingState':
        """Load state from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        state = cls()
        state.epoch = data.get('epoch', 0)
        state.total_epochs = data.get('total_epochs', 100)
        state.current_loss = data.get('current_loss', 0.0)
        state.losses = data.get('losses', state.losses)
        
        return state





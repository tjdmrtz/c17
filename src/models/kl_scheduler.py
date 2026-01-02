"""
KL Divergence Scheduling for VAE Training.

Implements various beta scheduling strategies:
1. Linear: Gradual increase from 0 to target
2. Cyclical: Periodic increase/decrease (good for disentanglement)
3. Warmup: Stay at 0 for N epochs, then increase
4. Monotonic: Only increase, never decrease

The goal is to first let the autoencoder learn to reconstruct well,
then gradually enforce latent space organization.

Usage:
    scheduler = KLScheduler(
        schedule_type='warmup',
        target_beta=1.0,
        warmup_epochs=20,
        anneal_epochs=30
    )
    
    for epoch in range(100):
        beta = scheduler.get_beta(epoch)
        # Use beta in VAE loss: loss = recon + beta * kl
"""

import math
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class KLSchedulerConfig:
    """Configuration for KL scheduler."""
    schedule_type: Literal['constant', 'linear', 'warmup', 'cyclical', 'sigmoid'] = 'warmup'
    target_beta: float = 1.0
    min_beta: float = 0.0
    warmup_epochs: int = 10  # Epochs with beta=0
    anneal_epochs: int = 20  # Epochs to reach target
    cycle_epochs: int = 10  # For cyclical: period of one cycle
    num_cycles: int = 4  # For cyclical: number of cycles


class KLScheduler:
    """
    Scheduler for KL divergence weight (beta) in VAE training.
    
    Strategies:
    - constant: Fixed beta throughout training
    - linear: Linear increase from min to target
    - warmup: Stay at min for warmup_epochs, then linear increase
    - cyclical: Periodic increase/decrease (helps disentanglement)
    - sigmoid: Smooth S-curve transition
    """
    
    def __init__(self, 
                 schedule_type: str = 'warmup',
                 target_beta: float = 1.0,
                 min_beta: float = 0.0,
                 warmup_epochs: int = 10,
                 anneal_epochs: int = 20,
                 cycle_epochs: int = 10,
                 num_cycles: int = 4,
                 total_epochs: Optional[int] = None):
        """
        Initialize KL scheduler.
        
        Args:
            schedule_type: Type of schedule ('constant', 'linear', 'warmup', 'cyclical', 'sigmoid')
            target_beta: Target/maximum beta value
            min_beta: Minimum beta value (usually 0)
            warmup_epochs: Epochs to stay at min_beta (for 'warmup')
            anneal_epochs: Epochs to transition from min to target
            cycle_epochs: Period of one cycle (for 'cyclical')
            num_cycles: Number of cycles (for 'cyclical')
            total_epochs: Total training epochs (for 'cyclical' auto-calculation)
        """
        self.schedule_type = schedule_type
        self.target_beta = target_beta
        self.min_beta = min_beta
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.cycle_epochs = cycle_epochs
        self.num_cycles = num_cycles
        self.total_epochs = total_epochs
        
        self.current_beta = min_beta
        self.history = []
    
    def get_beta(self, epoch: int, step: Optional[int] = None, 
                 total_steps: Optional[int] = None) -> float:
        """
        Get beta value for current epoch/step.
        
        Args:
            epoch: Current epoch (0-indexed)
            step: Current step within epoch (optional, for finer control)
            total_steps: Total steps per epoch (optional)
        
        Returns:
            Beta value for this epoch/step
        """
        if self.schedule_type == 'constant':
            beta = self.target_beta
        
        elif self.schedule_type == 'linear':
            # Linear increase over anneal_epochs
            if epoch >= self.anneal_epochs:
                beta = self.target_beta
            else:
                progress = epoch / self.anneal_epochs
                beta = self.min_beta + progress * (self.target_beta - self.min_beta)
        
        elif self.schedule_type == 'warmup':
            # Stay at min for warmup, then linear increase
            if epoch < self.warmup_epochs:
                beta = self.min_beta
            elif epoch < self.warmup_epochs + self.anneal_epochs:
                progress = (epoch - self.warmup_epochs) / self.anneal_epochs
                beta = self.min_beta + progress * (self.target_beta - self.min_beta)
            else:
                beta = self.target_beta
        
        elif self.schedule_type == 'cyclical':
            # Cyclical annealing (Bowman et al., Fu et al.)
            # Beta increases within each cycle, resets at start of new cycle
            if self.total_epochs:
                cycle_length = self.total_epochs / self.num_cycles
            else:
                cycle_length = self.cycle_epochs
            
            cycle_position = epoch % cycle_length
            progress = cycle_position / (cycle_length * 0.5)  # Ramp up in first half
            progress = min(1.0, progress)
            beta = self.min_beta + progress * (self.target_beta - self.min_beta)
        
        elif self.schedule_type == 'sigmoid':
            # Smooth S-curve transition
            if epoch < self.warmup_epochs:
                beta = self.min_beta
            else:
                # Sigmoid centered at warmup + anneal/2
                center = self.warmup_epochs + self.anneal_epochs / 2
                steepness = 10 / self.anneal_epochs  # Controls transition speed
                sigmoid = 1 / (1 + math.exp(-steepness * (epoch - center)))
                beta = self.min_beta + sigmoid * (self.target_beta - self.min_beta)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        self.current_beta = beta
        self.history.append(beta)
        return beta
    
    def get_beta_for_step(self, epoch: int, step: int, steps_per_epoch: int) -> float:
        """
        Get beta for a specific step (finer granularity than epoch).
        
        Args:
            epoch: Current epoch
            step: Current step within epoch
            steps_per_epoch: Total steps per epoch
        
        Returns:
            Beta value
        """
        # Convert to fractional epoch
        fractional_epoch = epoch + step / steps_per_epoch
        return self.get_beta(fractional_epoch)
    
    def __repr__(self):
        return (f"KLScheduler(type={self.schedule_type}, "
                f"target={self.target_beta}, "
                f"warmup={self.warmup_epochs}, "
                f"anneal={self.anneal_epochs})")


def visualize_schedules(total_epochs: int = 100, save_path: Optional[str] = None):
    """Visualize different KL scheduling strategies."""
    import matplotlib.pyplot as plt
    
    schedules = {
        'constant': KLScheduler('constant', target_beta=1.0),
        'linear': KLScheduler('linear', target_beta=1.0, anneal_epochs=50),
        'warmup': KLScheduler('warmup', target_beta=1.0, warmup_epochs=20, anneal_epochs=30),
        'cyclical': KLScheduler('cyclical', target_beta=1.0, total_epochs=total_epochs, num_cycles=4),
        'sigmoid': KLScheduler('sigmoid', target_beta=1.0, warmup_epochs=20, anneal_epochs=40),
    }
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    
    for (name, scheduler), color in zip(schedules.items(), colors):
        betas = [scheduler.get_beta(e) for e in range(total_epochs)]
        ax.plot(betas, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('Beta (KL weight)', color='white', fontsize=12)
    ax.set_title('KL Annealing Schedules', color='white', fontsize=14)
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_color('#3d3d5c')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("KL Scheduler Demo")
    print("=" * 50)
    
    # Demo each schedule
    schedules = [
        ('constant', KLScheduler('constant', target_beta=1.0)),
        ('linear', KLScheduler('linear', target_beta=1.0, anneal_epochs=50)),
        ('warmup', KLScheduler('warmup', target_beta=1.0, warmup_epochs=20, anneal_epochs=30)),
        ('cyclical', KLScheduler('cyclical', target_beta=1.0, total_epochs=100, num_cycles=4)),
        ('sigmoid', KLScheduler('sigmoid', target_beta=1.0, warmup_epochs=20, anneal_epochs=40)),
    ]
    
    print("\nBeta values at different epochs:")
    print("-" * 60)
    print(f"{'Schedule':<12} | Epoch 0 | Epoch 25 | Epoch 50 | Epoch 75 | Epoch 100")
    print("-" * 60)
    
    for name, scheduler in schedules:
        vals = [scheduler.get_beta(e) for e in [0, 25, 50, 75, 99]]
        print(f"{name:<12} | {vals[0]:>7.3f} | {vals[1]:>8.3f} | {vals[2]:>8.3f} | {vals[3]:>8.3f} | {vals[4]:>9.3f}")
    
    print("\n✅ Use visualize_schedules() to see plots")
    print("\nRecommended for your case:")
    print("  KLScheduler('warmup', target_beta=0.5, warmup_epochs=30, anneal_epochs=50)")
    print("  - First 30 epochs: focus on reconstruction (beta=0)")
    print("  - Epochs 30-80: gradually increase KL weight")
    print("  - After 80: full KL regularization")







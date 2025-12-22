"""
NiceGUI Dashboard for Neural ODE Training Monitoring.
"""

from .training_state import TrainingState
from .theme import THEME, GROUP_COLORS

# Dashboard requires nicegui - import conditionally
try:
    from .dashboard import TrainingDashboard, run_dashboard
    HAS_NICEGUI = True
except ImportError:
    TrainingDashboard = None
    run_dashboard = None
    HAS_NICEGUI = False

__all__ = ['TrainingState', 'TrainingDashboard', 'run_dashboard', 'THEME', 'GROUP_COLORS', 'HAS_NICEGUI']

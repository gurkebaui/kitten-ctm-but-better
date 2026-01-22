"""
Continuous Thought Machine (CTM) Package

A PyTorch implementation of the CTM architecture from Sakana AI.
"""

from .config import CTMConfig
from .neuron_level_models import NeuronLevelModels
from .synapse_model import SynapseModel, SimpleSynapseModel
from .synchronization import SynchronizationModule, DualSynchronization
from .machine import ContinuousThoughtMachine, CTMWithBackbone
from .loss import CTMLoss, CTMLossWithRegularization
from .trainer import CTMTrainer, create_optimizer, create_scheduler


__version__ = "0.1.0"

__all__ = [
    # Config
    "CTMConfig",
    
    # Core Components
    "NeuronLevelModels",
    "SynapseModel",
    "SimpleSynapseModel",
    "SynchronizationModule",
    "DualSynchronization",
    
    # Main Model
    "ContinuousThoughtMachine",
    "CTMWithBackbone",
    
    # Loss
    "CTMLoss",
    "CTMLossWithRegularization",
    
    # Training
    "CTMTrainer",
    "create_optimizer",
    "create_scheduler",
]
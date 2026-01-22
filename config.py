from dataclasses import dataclass
from typing import Optional

@dataclass
class CTMConfig:
    # Core Architecture
    D: int = 4                   # Model width
    M: int = 2                    # kürzere Neuron-History
    T_max: int = 10               # weniger interne Ticks
    
    # Input / Embeddings
    d_embed: int = 4             # MUSS zu D passen
    tokenizer_name: str = "gpt2"
    max_context_len: int = 4096   # realistischer für kleines Modell
    num_heads: int = 2            # 64 / 4 = 16 -> sauber
    
    # Neuron-Level Models
    nlm_hidden_dim: int = 4       # vorher viel zu fett
    nlm_num_layers: int = 1       # reicht locker bei D=64
    
    # Synapse Model
    synapse_depth: int = 3
    synapse_hidden_mult: float = 1.5
    synapse_dropout: float = 0.1
    
    # Synchronization
    D_action: int = 2            # max <= D
    D_output: int = 2
    sync_eps: float = 1e-8
    
    # Output
    num_classes: int = 1
    
    # Device
    device: str = "cuda"
from dataclasses import dataclass
import torch

@dataclass
class CTMConfig:
    # --- Input / Embedding ---
    d_embed_input: int = 384  # Output Dim von all-MiniLM-L6-v2
    max_context_len: int = 512 
    
    # --- CTM Core Architecture ---
    D: int = 256              # Interne Breite (256 ist effizient & stark für binary)
    M: int = 16               # Neuron Memory Länge
    T_max: int = 30           # Anzahl der Denk-Schritte (Ticks)
    
    # --- Components ---
    nlm_hidden_dim: int = 32
    nlm_num_layers: int = 2
    
    # Synapse (Informationsmischung)
    synapse_depth: int = 3
    synapse_dropout: float = 0.1
    
    # Attention
    num_heads: int = 4        # 4 Heads reichen bei D=256
    
    # --- Synchronization & Output ---
    D_action: int = 128
    D_output: int = 128
    num_classes: int = 1      # 1 = Binary (Sigmoid), 2+ = Softmax
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
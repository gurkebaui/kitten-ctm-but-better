import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CTMConfig

# --- 1. Neuron Level Models (NLMs) ---
class NeuronLevelModels(nn.Module):
    """Verarbeitet die Historie jedes Neurons individuell."""
    def __init__(self, D, M, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        # Input: M -> hidden. Groups=D bedeutet: D unabhängige Netzwerke
        self.layers.append(nn.Conv1d(D, D * hidden_dim, kernel_size=M, groups=D))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv1d(D * hidden_dim, D * hidden_dim, kernel_size=1, groups=D))
        # Output: hidden -> 1
        self.layers.append(nn.Conv1d(D * hidden_dim, D, kernel_size=1, groups=D))
        self.activation = nn.GELU()

    def forward(self, history):
        # history: [Batch, D, Memory]
        x = history
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x.squeeze(-1) # [Batch, D]

# --- 2. Synchronization ---
class SynchronizationModule(nn.Module):
    """Berechnet die Synchronisation zwischen zufälligen Neuronen-Paaren."""
    def __init__(self, D, D_sample):
        super().__init__()
        # Wähle feste zufällige Paare
        idx_i = torch.randint(0, D, (D_sample,))
        idx_j = torch.randint(0, D, (D_sample,))
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)
        self.decay = nn.Parameter(torch.zeros(D_sample)) # Lernbarer Zeit-Decay

    def forward(self, z_hist):
        # z_hist: [Batch, Ticks, D]
        B, T, D = z_hist.shape
        z_i = z_hist[:, :, self.idx_i]
        z_j = z_hist[:, :, self.idx_j]
        
        # Zeitlicher Decay (Exponentiell)
        times = torch.arange(T, device=z_hist.device).flip(0)
        weights = torch.exp(-F.softplus(self.decay) * times.unsqueeze(1))
        
        # Gewichtetes inneres Produkt
        num = (z_i * z_j * weights.unsqueeze(0)).sum(dim=1)
        den = torch.sqrt((weights**2).sum(dim=0) + 1e-8)
        return num / den

class DualSynchronization(nn.Module):
    def __init__(self, D, D_action, D_output):
        super().__init__()
        self.action_sync = SynchronizationModule(D, D_action)
        self.output_sync = SynchronizationModule(D, D_output)

    def forward(self, z_hist):
        return self.action_sync(z_hist), self.output_sync(z_hist)

# --- 3. Main CTM Class ---
class ContinuousThoughtMachine(nn.Module):
    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config
        self.D = config.D
        
        # Projektion: Input Embeddings (384) -> Interne Größe (256)
        self.input_proj = nn.Sequential(
            nn.Linear(config.d_embed_input, config.D),
            nn.LayerNorm(config.D),
            nn.GELU()
        )
        
        # Komponenten
        self.nlms = NeuronLevelModels(config.D, config.M, config.nlm_hidden_dim, config.nlm_num_layers)
        
        # Synapse: Mischt Gedanken (z) und Wahrnehmung (o)
        self.synapse = nn.Sequential(
            nn.Linear(config.D * 2, config.D * 2),
            nn.GELU(),
            nn.Dropout(config.synapse_dropout),
            nn.Linear(config.D * 2, config.D)
        )
        
        self.sync = DualSynchronization(config.D, config.D_action, config.D_output)
        
        # Attention (schaut auf den Kontext)
        self.attn = nn.MultiheadAttention(embed_dim=config.D, num_heads=config.num_heads, batch_first=True)
        self.query_proj = nn.Linear(config.D_action, config.D)
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(config.D_output, config.D_output),
            nn.GELU(),
            nn.Linear(config.D_output, config.num_classes)
        )
        
        # Lernbare Start-Zustände
        self.z_init = nn.Parameter(torch.randn(1, config.D) * 0.02)
        self.hist_init = nn.Parameter(torch.zeros(1, config.D, config.M))

    def _certainty(self, logits):
        """Berechnet Sicherheit (1 - Entropie) für binary classification"""
        probs = torch.sigmoid(logits)
        eps = 1e-9
        entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        # Normieren auf 0..1 (Max Entropie bei 0.5 ist ~0.693)
        return 1.0 - (entropy / 0.693147)

    def forward(self, embeddings, return_all=False):
        """
        embeddings: [Batch, Seq, 384] (Vorberechnet!)
        """
        B = embeddings.shape[0]
        device = embeddings.device
        
        # 1. Input Projizieren
        ctx = self.input_proj(embeddings) # [B, Seq, D]
        
        # 2. Init State
        z = self.z_init.expand(B, -1)
        hist = self.hist_init.expand(B, -1, -1) # [B, D, M]
        z_history = z.unsqueeze(1)               # [B, 1, D]
        
        all_logits = []
        
        # 3. Thinking Loop
        for t in range(self.config.T_max):
            # A. Sync
            s_act, s_out = self.sync(z_history)
            
            # B. Attention (Input anschauen)
            query = self.query_proj(s_act).unsqueeze(1)
            # Da wir Embeddings haben, brauchen wir meist keine Maske bei Batch=1
            attn_out, _ = self.attn(query, ctx, ctx)
            attn_out = attn_out.squeeze(1)
            
            # C. Synapse (Mixen)
            combined = torch.cat([z, attn_out], dim=-1)
            pre_act = self.synapse(combined) # [B, D]
            
            # D. Historie updaten (FIFO: ältester raus, neuer rein)
            hist = torch.cat([hist[:, :, 1:], pre_act.unsqueeze(-1)], dim=-1)
            
            # E. NLM (Neues Feuern der Neuronen)
            z = self.nlms(hist)
            z_history = torch.cat([z_history, z.unsqueeze(1)], dim=1)
            
            # F. Prediction
            logits = self.head(s_out)
            all_logits.append(logits)
            
        # 4. Resultate aggregieren
        all_logits = torch.stack(all_logits, dim=1) # [B, T, 1]
        certainties = self._certainty(all_logits)
        
        # Adaptive Compute: Wähle Tick mit höchster Sicherheit
        best_tick = certainties.argmax(dim=1)
        batch_idx = torch.arange(B, device=device)
        
        final_logits = all_logits[batch_idx, best_tick.squeeze()].squeeze(-1)
        final_probs = torch.sigmoid(final_logits)
        
        return {
            "logits": final_logits,
            "probs": final_probs,
            "all_logits": all_logits,
            "certainties": certainties,
            "ticks_used": best_tick,
            "z_history": z_history if return_all else None
        }
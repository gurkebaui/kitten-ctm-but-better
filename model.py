import torch
import torch.nn as nn
import math
from typing import Dict

# Import your specialized modules
from neuron_level_models import NeuronLevelModels
from synapse_model import SynapseModel
from synchronization import DualSynchronization
from loss import CTMLoss
from config import CTMConfig

class ContinuousThoughtMachine(nn.Module):
    def __init__(self, config: CTMConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.D = config.D
        self.M = config.M
        self.T_max = config.T_max
        self.d_embed = config.d_embed
        
        # 1. Embedding Layer (Missing in original, needed for text)
        self.embedding = nn.Embedding(vocab_size, self.d_embed)
        
        # 2. Neuron-Level Models (From neuron_level_models.py)
        self.nlms = NeuronLevelModels(
            D=config.D, 
            M=config.M, 
            hidden_dim=config.nlm_hidden_dim, 
            num_layers=config.nlm_num_layers
        )
        
        # 3. Synapse Model (From synapse_model.py)
        # Input: Concat of latent state z + attention output o
        self.synapse = SynapseModel(
            input_dim=config.D + config.d_embed,
            output_dim=config.D,
            depth=config.synapse_depth,
            hidden_mult=config.synapse_hidden_mult,
            dropout=config.synapse_dropout
        )
        
        # 4. Synchronization (From synchronization.py)
        self.sync = DualSynchronization(
            D=config.D,
            D_action=config.D_action,
            D_output=config.D_output,
            eps=config.sync_eps
        )
        
        # 5. Attention Mechanism
        # Cross attention: Query from sync, Key/Value from Embeddings
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_embed,
            num_heads=config.num_heads if hasattr(config, 'num_heads') else 8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projections
        self.q_proj = nn.Sequential(
            nn.Linear(config.D_action, config.d_embed),
            nn.LayerNorm(config.d_embed)
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(config.D_output, config.D_output // 2),
            nn.GELU(),
            nn.LayerNorm(config.D_output // 2),
            nn.Linear(config.D_output // 2, config.num_classes)
        )
        
        # Initialization parameters
        self.z_init = nn.Parameter(torch.randn(1, config.D) * 0.02)
        self.hist_init = nn.Parameter(torch.randn(1, config.D, config.M) * 0.02)

    def _certainty(self, p):
        """Calculates certainty based on entropy of prediction."""
        eps = 1e-8
        if p.dim() == 3 and p.shape[-1] == 1:
            p = p.squeeze(-1)
        if p.dim() == 2:
            # Binary entropy
            ent = -(p * torch.log2(p + eps) + (1 - p) * torch.log2(1 - p + eps))
            return 1 - ent
        # Multi-class entropy (not used in binary case)
        ent = -(p * torch.log2(p + eps)).sum(-1)
        return 1 - ent / math.log2(p.shape[-1])

    def forward(self, input_ids, attention_mask=None, return_all=False):
        """
        input_ids: [Batch, Seq_Len]
        """
        B = input_ids.shape[0]
        device = input_ids.device
        
        # 1. Embed inputs
        # ctx: [Batch, Seq_Len, D_embed]
        ctx = self.embedding(input_ids)
        
        # 2. Initialize State
        z = self.z_init.expand(B, -1).clone()
        hist = self.hist_init.expand(B, -1, -1).clone()
        z_hist = z.unsqueeze(1) # [B, 1, D]
        
        preds = []
        
        # 3. Unfold Time (Internal Ticks)
        for _ in range(self.T_max):
            # Compute synchronization from history
            s_act, s_out = self.sync(z_hist)
            
            # Generate Query from Action Sync
            q = self.q_proj(s_act).unsqueeze(1) # [B, 1, D_embed]
            
            # Cross Attention: Query (state) attends to Context (data)
            # o: [B, 1, D_embed]
            o, w = self.attn(
                query=q, 
                key=ctx, 
                value=ctx, 
                key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
            )
            o = o.squeeze(1)
            
            # Update State via Synapse Model
            # Concat prev state z and attention o
            a = self.synapse(torch.cat([z, o], -1)) # [B, D]
            
            # Update history
            hist = torch.cat([hist[:, :, 1:], a.unsqueeze(-1)], -1)
            
            # Update Neurons via NLMs
            z = self.nlms(hist)
            
            # Update full history for next sync calculation
            z_hist = torch.cat([z_hist, z.unsqueeze(1)], 1)
            
            # Prediction
            pred = self.out_proj(s_out) # [B, 1]
            pred = torch.sigmoid(pred)
            preds.append(pred)
            
        preds = torch.stack(preds, 1) # [B, T, 1]
        certs = self._certainty(preds) # [B, T]
        
        # Select final prediction based on highest certainty (Adaptive Compute)
        idx = certs.argmax(1)
        batch_idx = torch.arange(B, device=device)
        final = preds[batch_idx, idx] # [B, 1]
        
        output = {
            'predictions': preds,
            'certainties': certs,
            'final_prediction': final
        }
        
        if return_all:
            output['z_history'] = z_hist
            
        return output
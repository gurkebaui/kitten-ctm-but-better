import torch
import torch.nn as nn
import torch.nn.functional as F

class CTMLoss(nn.Module):
    def __init__(self, aux_weight=0.1):
        super().__init__()
        self.aux_weight = aux_weight # Gewicht für den Loss über alle Ticks
    
    def forward(self, all_logits, certainties, targets):
        """
        all_logits: [Batch, T, 1]
        certainties: [Batch, T, 1]
        targets: [Batch]
        """
        B, T, _ = all_logits.shape
        
        # Target expandieren für Zeitdimension
        targets_exp = targets.unsqueeze(1).unsqueeze(2).expand(B, T, 1)
        
        # BCE Loss pro Tick berechnen (Reduction=None um pro Tick zu sehen)
        bce_per_tick = F.binary_cross_entropy_with_logits(all_logits, targets_exp, reduction='none') # [B, T, 1]
        
        # 1. Finde t1: Der Tick mit dem kleinsten Loss (Beste Vorhersage)
        loss_vals = bce_per_tick.squeeze(-1)
        t1_idx = loss_vals.argmin(dim=1)
        
        # 2. Finde t2: Der Tick mit der höchsten Sicherheit
        cert_vals = certainties.squeeze(-1)
        t2_idx = cert_vals.argmax(dim=1)
        
        # Sammle die Losses an t1 und t2
        batch_idx = torch.arange(B, device=all_logits.device)
        loss_t1 = loss_vals[batch_idx, t1_idx]
        loss_t2 = loss_vals[batch_idx, t2_idx]
        
        # Hauptloss: Durchschnitt
        primary_loss = (loss_t1 + loss_t2) / 2.0
        
        # Aux Loss: Ein bisschen Loss auf alle Ticks, damit das Modell allgemein lernt
        aux_loss = loss_vals.mean(dim=1)
        
        total_loss = primary_loss + self.aux_weight * aux_loss
        
        return total_loss.mean()
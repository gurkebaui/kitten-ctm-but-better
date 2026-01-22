"""
CTM Loss Function Module

Implements the unique dynamic loss function that enables
adaptive computation in the CTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CTMLoss(nn.Module):
    """
    Dynamic CTM Loss Function.
    
    The CTM loss aggregates losses from two specific ticks:
    - t1: The tick with minimum loss (best prediction)
    - t2: The tick with maximum certainty (most confident prediction)
    
    Final Loss = (L_t1 + L_t2) / 2
    
    This encourages the model to:
    1. Develop meaningful computations across multiple ticks
    2. Learn when to be confident
    3. Adaptively use more computation for harder examples
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        label_smoothing: float = 0.0,
        auxiliary_weight: float = 0.0
    ):
        """
        Initialize the CTM loss.
        
        Args:
            num_classes: Number of output classes (1 for binary)
            label_smoothing: Label smoothing factor
            auxiliary_weight: Weight for auxiliary loss (all-tick average)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.auxiliary_weight = auxiliary_weight
    
    def _compute_per_tick_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss at each tick.
        
        Args:
            predictions: [batch_size, T, num_classes] or [batch_size, T]
            targets: [batch_size] or [batch_size, num_classes]
        
        Returns:
            per_tick_loss: [batch_size, T]
        """
        batch_size, T = predictions.shape[:2]
        
        # Ensure targets are 1D: [B]
        targets = targets.view(batch_size)
        
        if self.num_classes == 1:
            # Binary classification
            if predictions.dim() == 3:
                predictions = predictions.squeeze(-1)  # [B, T]
            
            # Expand targets to match predictions
            targets_expanded = targets.unsqueeze(1).expand(-1, T)  # [B, T]
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                targets_expanded = targets_expanded * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Binary cross-entropy
            per_tick_loss = F.binary_cross_entropy(
                predictions,
                targets_expanded.float(),
                reduction='none'
            )  # [B, T]
        else:
            # Multi-class classification
            targets_expanded = targets.unsqueeze(1).expand(-1, T)  # [B, T]
            
            predictions_flat = predictions.view(-1, self.num_classes)  # [B*T, C]
            targets_flat = targets_expanded.reshape(-1).long()  # [B*T]
            
            loss_flat = F.cross_entropy(
                predictions_flat,
                targets_flat,
                reduction='none',
                label_smoothing=self.label_smoothing
            )
            
            per_tick_loss = loss_flat.view(batch_size, T)
        
        return per_tick_loss
    
    def forward(
        self,
        predictions: torch.Tensor,
        certainties: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute the dynamic CTM loss.
        
        Args:
            predictions: Predictions at each tick [batch_size, T, num_classes]
            certainties: Certainties at each tick [batch_size, T]
            targets: Ground truth labels [batch_size]
            return_components: Whether to return loss components
        
        Returns:
            loss: Scalar loss value
            (optional) components: Dictionary with loss components
        """
        batch_size, T = predictions.shape[:2]
        device = predictions.device
        
        # Compute loss at each tick
        per_tick_loss = self._compute_per_tick_loss(predictions, targets)  # [B, T]
        
        # Find t1: tick with minimum loss for each sample
        min_loss_values, t1_indices = torch.min(per_tick_loss, dim=1)  # [B], [B]
        
        # Find t2: tick with maximum certainty for each sample
        max_cert_values, t2_indices = torch.max(certainties, dim=1)  # [B], [B]
        
        # Gather losses at t1 and t2
        batch_indices = torch.arange(batch_size, device=device)
        loss_at_t1 = per_tick_loss[batch_indices, t1_indices]  # [B]
        loss_at_t2 = per_tick_loss[batch_indices, t2_indices]  # [B]
        
        # Primary loss: average of t1 and t2 losses
        primary_loss = (loss_at_t1 + loss_at_t2) / 2
        
        # Optional auxiliary loss: average across all ticks
        if self.auxiliary_weight > 0:
            auxiliary_loss = per_tick_loss.mean(dim=1)  # [B]
            total_loss = primary_loss + self.auxiliary_weight * auxiliary_loss
        else:
            total_loss = primary_loss
        
        # Reduce across batch
        final_loss = total_loss.mean()
        
        if return_components:
            components = {
                'primary_loss': primary_loss.mean(),
                'loss_at_t1': loss_at_t1.mean(),
                'loss_at_t2': loss_at_t2.mean(),
                'avg_t1': t1_indices.float().mean(),
                'avg_t2': t2_indices.float().mean(),
                'avg_certainty': certainties.mean(),
                'max_certainty': max_cert_values.mean(),
                'per_tick_loss': per_tick_loss.mean(dim=0)  # [T]
            }
            if self.auxiliary_weight > 0:
                components['auxiliary_loss'] = auxiliary_loss.mean()
            return final_loss, components
        
        return final_loss


class CTMLossWithRegularization(CTMLoss):
    """
    CTM Loss with additional regularization terms.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        label_smoothing: float = 0.0,
        auxiliary_weight: float = 0.0,
        certainty_reg_weight: float = 0.01,
        sync_diversity_weight: float = 0.01
    ):
        super().__init__(num_classes, label_smoothing, auxiliary_weight)
        
        self.certainty_reg_weight = certainty_reg_weight
        self.sync_diversity_weight = sync_diversity_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        certainties: torch.Tensor,
        targets: torch.Tensor,
        z_history: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute loss with regularization.
        
        Additional Args:
            z_history: Post-activation history [batch_size, T+1, D]
                      Used for computing synchronization diversity regularization
        """
        # Get base loss
        if return_components:
            base_loss, components = super().forward(
                predictions, certainties, targets, return_components=True
            )
        else:
            base_loss = super().forward(predictions, certainties, targets)
            components = {}
        
        total_loss = base_loss
        
        # Certainty regularization: encourage certainty to increase over time
        if self.certainty_reg_weight > 0:
            # Penalize decreasing certainty
            cert_diff = certainties[:, 1:] - certainties[:, :-1]  # [B, T-1]
            cert_reg = F.relu(-cert_diff).mean()  # Penalize decreases
            total_loss = total_loss + self.certainty_reg_weight * cert_reg
            components['certainty_reg'] = cert_reg
        
        # Synchronization diversity: encourage diverse neural activity
        if self.sync_diversity_weight > 0 and z_history is not None:
            # Compute variance of z across time
            z_var = z_history.var(dim=1).mean()  # Mean variance across neurons
            # Penalize low variance (encourage temporal dynamics)
            diversity_reg = 1.0 / (z_var + 1e-6)
            total_loss = total_loss + self.sync_diversity_weight * diversity_reg
            components['diversity_reg'] = diversity_reg
        
        if return_components:
            return total_loss, components
        return total_loss
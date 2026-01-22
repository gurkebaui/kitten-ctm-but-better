"""
Neural Synchronization Module

Computes the synchronization matrix that captures temporal correlations
between neurons. This is the CTM's primary latent representation.

Key Concept: The model's understanding is based on how neurons fire
together over time - inspired by Hebbian learning ("neurons that fire
together, wire together").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SynchronizationModule(nn.Module):
    """
    Computes time-decayed neural synchronization vectors.
    
    The synchronization between neurons i and j is computed as:
    S_ij = (Σ_τ exp(-r_ij(t-τ)) * z_i^τ * z_j^τ) / sqrt(Σ_τ exp(-r_ij(t-τ)))
    
    Where:
    - z_i^τ is the post-activation of neuron i at time τ
    - r_ij is a learnable decay rate for the pair (i,j)
    - t is the current time step
    
    The decay allows the model to focus on different time scales
    for different neuron pairs.
    """
    
    def __init__(
        self,
        D: int,
        D_sample: int,
        eps: float = 1e-8,
        init_decay: float = 0.0
    ):
        """
        Initialize the synchronization module.
        
        Args:
            D: Number of neurons (model width)
            D_sample: Number of neuron pairs to sample
            eps: Small constant for numerical stability
            init_decay: Initial value for decay parameters
        """
        super().__init__()
        
        self.D = D
        self.D_sample = D_sample
        self.eps = eps
        
        # Learnable decay rates (one per neuron pair)
        # Initialized to zero = no decay (equal weighting across time)
        self.decay_rates = nn.Parameter(
            torch.full((D_sample,), init_decay)
        )
        
        # Pre-compute random indices for neuron pair sampling
        # Register as buffer so they're saved with the model but not trained
        indices_i, indices_j = self._generate_pair_indices(D, D_sample)
        self.register_buffer('indices_i', indices_i)
        self.register_buffer('indices_j', indices_j)
        
        # Cache for recursive computation
        self._cached_numerator = None
        self._cached_denominator = None
        self._cache_valid = False
    
    def _generate_pair_indices(
        self,
        D: int,
        D_sample: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random indices for neuron pair sampling.
        
        Ensures i != j to avoid self-synchronization.
        """
        # Generate more pairs than needed, then filter
        indices_i = torch.randint(0, D, (D_sample * 2,))
        indices_j = torch.randint(0, D, (D_sample * 2,))
        
        # Filter out pairs where i == j
        valid_mask = indices_i != indices_j
        indices_i = indices_i[valid_mask][:D_sample]
        indices_j = indices_j[valid_mask][:D_sample]
        
        # If we don't have enough, just regenerate (rare edge case)
        while len(indices_i) < D_sample:
            new_i = torch.randint(0, D, (D_sample,))
            new_j = torch.randint(0, D, (D_sample,))
            valid_mask = new_i != new_j
            indices_i = torch.cat([indices_i, new_i[valid_mask]])[:D_sample]
            indices_j = torch.cat([indices_j, new_j[valid_mask]])[:D_sample]
        
        return indices_i.long(), indices_j.long()
    
    def reset_cache(self):
        """Reset the recursive computation cache."""
        self._cached_numerator = None
        self._cached_denominator = None
        self._cache_valid = False
    
    def forward(
        self,
        z_history: torch.Tensor,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Compute the synchronization vector.
        
        Args:
            z_history: Post-activation history of shape [batch_size, t, D]
                where t is the number of time steps so far
            use_cache: Whether to use recursive computation for efficiency
        
        Returns:
            sync_vector: Synchronization vector of shape [batch_size, D_sample]
        """
        batch_size, t, D = z_history.shape
        device = z_history.device
        
        # Get decay rates (ensure non-negative via softplus)
        r = F.softplus(self.decay_rates)  # [D_sample]
        
        # Gather neuron histories for selected pairs
        z_i = z_history[:, :, self.indices_i]  # [B, t, D_sample]
        z_j = z_history[:, :, self.indices_j]  # [B, t, D_sample]
        
        # Compute time decay weights
        # For each time step τ, weight = exp(-r * (t-1-τ))
        # where τ=0 is oldest, τ=t-1 is newest
        time_indices = torch.arange(t, device=device).float()  # [t]
        decay_weights = torch.exp(
            -r.unsqueeze(0) * (t - 1 - time_indices).unsqueeze(1)
        )  # [t, D_sample]
        
        # Compute weighted products
        products = z_i * z_j  # [B, t, D_sample]
        
        # Numerator: sum of weighted products
        weighted_products = products * decay_weights.unsqueeze(0)  # [B, t, D_sample]
        numerator = weighted_products.sum(dim=1)  # [B, D_sample]
        
        # Denominator: sqrt of sum of weights
        denominator = torch.sqrt(decay_weights.sum(dim=0) + self.eps)  # [D_sample]
        
        # Compute synchronization
        sync_vector = numerator / denominator.unsqueeze(0)  # [B, D_sample]
        
        return sync_vector
    
    def forward_recursive(
        self,
        z_new: torch.Tensor,
        z_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Recursive computation of synchronization for efficiency.
        
        Instead of recomputing the full sum at each step, we maintain
        running sums and update them incrementally.
        
        Args:
            z_new: New post-activation of shape [batch_size, D]
            z_history: Full history (used only for initialization)
        
        Returns:
            sync_vector: Synchronization vector of shape [batch_size, D_sample]
        """
        batch_size = z_new.shape[0]
        device = z_new.device
        
        # Get decay rates
        r = F.softplus(self.decay_rates)  # [D_sample]
        decay_factor = torch.exp(-r)  # [D_sample]
        
        # Get new neuron values for selected pairs
        z_i_new = z_new[:, self.indices_i]  # [B, D_sample]
        z_j_new = z_new[:, self.indices_j]  # [B, D_sample]
        new_product = z_i_new * z_j_new  # [B, D_sample]
        
        if not self._cache_valid:
            # Initialize from full history if provided
            if z_history is not None:
                sync = self.forward(z_history, use_cache=False)
                # Initialize cache (would need more complex bookkeeping)
                self._cached_numerator = new_product
                self._cached_denominator = torch.ones(
                    self.D_sample, device=device
                )
            else:
                self._cached_numerator = new_product
                self._cached_denominator = torch.ones(
                    self.D_sample, device=device
                )
            self._cache_valid = True
        else:
            # Recursive update:
            # new_numerator = decay_factor * old_numerator + new_product
            # new_denominator = decay_factor * old_denominator + 1
            self._cached_numerator = (
                decay_factor.unsqueeze(0) * self._cached_numerator + new_product
            )
            self._cached_denominator = (
                decay_factor * self._cached_denominator + 1.0
            )
        
        # Compute synchronization
        sync_vector = self._cached_numerator / (
            torch.sqrt(self._cached_denominator + self.eps).unsqueeze(0)
        )
        
        return sync_vector


class DualSynchronization(nn.Module):
    """
    Manages both action and output synchronization modules.
    
    The CTM uses two separate synchronization vectors:
    - S_action: Used to generate attention queries
    - S_output: Used to generate final predictions
    """
    
    def __init__(
        self,
        D: int,
        D_action: int,
        D_output: int,
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.action_sync = SynchronizationModule(D, D_action, eps)
        self.output_sync = SynchronizationModule(D, D_output, eps)
    
    def reset_cache(self):
        """Reset caches for both modules."""
        self.action_sync.reset_cache()
        self.output_sync.reset_cache()
    
    def forward(
        self,
        z_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both synchronization vectors.
        
        Args:
            z_history: Post-activation history [batch_size, t, D]
        
        Returns:
            s_action: Action synchronization [batch_size, D_action]
            s_output: Output synchronization [batch_size, D_output]
        """
        s_action = self.action_sync(z_history)
        s_output = self.output_sync(z_history)
        
        return s_action, s_output
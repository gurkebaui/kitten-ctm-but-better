"""
Neuron-Level Models (NLMs) Module

Each neuron in the CTM has its own private MLP that processes
the history of incoming pre-activations to determine its output.
This replaces traditional static activation functions like ReLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class NeuronLevelModels(nn.Module):
    """
    Implements D independent MLPs for D neurons in a parallelized manner.
    
    Each NLM takes an M-dimensional pre-activation history as input
    and produces a single scalar value: the post-activation for that neuron.
    
    Key Innovation: Instead of a simple activation function f(x), each neuron
    computes f(history of x), allowing temporal dynamics at the neuron level.
    
    Implementation uses batched matrix operations via einsum for efficiency,
    avoiding Python loops over individual neurons.
    """
    
    def __init__(
        self,
        D: int,
        M: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        init_scale: float = 0.02
    ):
        """
        Initialize the NLM module.
        
        Args:
            D: Number of neurons (model width)
            M: Memory length (history of pre-activations per neuron)
            hidden_dim: Hidden dimension of each neuron's private MLP
            num_layers: Number of layers in each NLM (minimum 2)
            init_scale: Scale for weight initialization
        """
        super().__init__()
        
        self.D = D
        self.M = M
        self.hidden_dim = hidden_dim
        self.num_layers = max(2, num_layers)
        self.init_scale = init_scale
        
        # Build layer weights and biases
        # Each neuron has its own set of weights (D independent MLPs)
        self.layer_weights = nn.ParameterList()
        self.layer_biases = nn.ParameterList()
        
        # Input layer: M -> hidden_dim (for each of D neurons)
        self.layer_weights.append(
            nn.Parameter(torch.empty(D, M, hidden_dim))
        )
        self.layer_biases.append(
            nn.Parameter(torch.zeros(D, hidden_dim))
        )
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(self.num_layers - 2):
            self.layer_weights.append(
                nn.Parameter(torch.empty(D, hidden_dim, hidden_dim))
            )
            self.layer_biases.append(
                nn.Parameter(torch.zeros(D, hidden_dim))
            )
        
        # Output layer: hidden_dim -> 1
        self.layer_weights.append(
            nn.Parameter(torch.empty(D, hidden_dim, 1))
        )
        self.layer_biases.append(
            nn.Parameter(torch.zeros(D, 1))
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using scaled initialization.
        
        Uses a combination of Xavier/Kaiming initialization
        scaled by init_scale for stability.
        """
        for i, (weight, bias) in enumerate(zip(self.layer_weights, self.layer_biases)):
            # Compute fan_in and fan_out for this layer
            fan_in = weight.shape[1]
            fan_out = weight.shape[2]
            
            # Xavier-like initialization with custom scaling
            std = self.init_scale * math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(weight, mean=0.0, std=std)
            nn.init.zeros_(bias)
    
    def forward(self, pre_activation_history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all D neurons simultaneously.
        
        Args:
            pre_activation_history: Tensor of shape [batch_size, D, M]
                containing the last M pre-activations for each neuron.
        
        Returns:
            post_activations: Tensor of shape [batch_size, D]
                containing the new state for each neuron.
        
        Implementation Details:
            Uses torch.einsum for efficient parallel matrix multiplication
            across all D neurons. The equation 'bdm,dmh->bdh' means:
            - b: batch dimension
            - d: neuron dimension (D independent MLPs)
            - m: input dimension (memory)
            - h: hidden dimension
        """
        batch_size = pre_activation_history.shape[0]
        
        # Current activation state
        x = pre_activation_history  # [B, D, M]
        
        # Pass through all layers
        for i, (weight, bias) in enumerate(zip(self.layer_weights, self.layer_biases)):
            # Determine einsum equation based on dimensions
            in_dim = weight.shape[1]
            out_dim = weight.shape[2]
            
            # Batched matrix multiplication for all D neurons
            # Each neuron applies its own weight matrix
            if i == 0:
                # First layer: [B, D, M] @ [D, M, H] -> [B, D, H]
                x = torch.einsum('bdm,dmh->bdh', x, weight)
            else:
                # Hidden/output layers: [B, D, H] @ [D, H, O] -> [B, D, O]
                x = torch.einsum('bdh,dho->bdo', x, weight)
            
            # Add bias (broadcast across batch)
            x = x + bias.unsqueeze(0)
            
            # Apply activation for all but the last layer
            if i < len(self.layer_weights) - 1:
                x = F.gelu(x)  # GELU often works better than ReLU
        
        # Squeeze output dimension: [B, D, 1] -> [B, D]
        return x.squeeze(-1)


class NeuronLevelModelsConv(nn.Module):
    """
    Alternative NLM implementation using 1D convolutions.
    
    This can be more memory-efficient for very large D,
    as it uses grouped convolutions instead of einsum.
    """
    
    def __init__(
        self,
        D: int,
        M: int,
        hidden_dim: int = 64,
        init_scale: float = 0.02
    ):
        super().__init__()
        
        self.D = D
        self.M = M
        self.hidden_dim = hidden_dim
        
        # Use grouped 1D convolutions where groups=D
        # This effectively creates D independent 1D conv networks
        
        # Layer 1: Process M-length history
        self.conv1 = nn.Conv1d(
            in_channels=D,
            out_channels=D * hidden_dim,
            kernel_size=M,
            groups=D,
            bias=True
        )
        
        # Layer 2: Hidden to output
        self.linear_out = nn.Conv1d(
            in_channels=D * hidden_dim,
            out_channels=D,
            kernel_size=1,
            groups=D,
            bias=True
        )
        
        self._initialize_weights(init_scale)
    
    def _initialize_weights(self, init_scale: float):
        for module in [self.conv1, self.linear_out]:
            nn.init.normal_(module.weight, std=init_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, pre_activation_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre_activation_history: [batch_size, D, M]
        
        Returns:
            post_activations: [batch_size, D]
        """
        # Apply first convolution across memory dimension
        x = self.conv1(pre_activation_history)  # [B, D*hidden, 1]
        x = F.gelu(x)
        
        # Apply output projection
        x = self.linear_out(x)  # [B, D, 1]
        
        return x.squeeze(-1)  # [B, D]
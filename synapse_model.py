"""
Synapse Model Module

The synapse model is a U-Net-style MLP that processes the concatenation
of attention output and previous post-activation state to produce
pre-activations for the NLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class UNetBlock(nn.Module):
    """
    A single block of the U-Net style MLP.
    
    Includes a linear layer, normalization, activation, and optional dropout.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_norm: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SynapseModel(nn.Module):
    """
    U-Net-style MLP for producing pre-activations.
    
    Architecture:
    1. Encoder: Progressively reduces dimension
    2. Bottleneck: Smallest representation
    3. Decoder: Progressively increases dimension with skip connections
    
    The skip connections allow gradients to flow more easily and
    help preserve information from the input.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 4,
        hidden_mult: float = 2.0,
        dropout: float = 0.1,
        init_scale: float = 0.02
    ):
        """
        Initialize the Synapse Model.
        
        Args:
            input_dim: Dimension of input (concat of attention output + prev state)
            output_dim: Dimension of output (D pre-activations)
            depth: Number of encoder/decoder layers
            hidden_mult: Multiplier for hidden dimension at each level
            dropout: Dropout rate
            init_scale: Scale for weight initialization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        
        # Calculate dimensions for each level
        # Start from output_dim and scale up, then back down
        dims = self._compute_dimensions(input_dim, output_dim, depth, hidden_mult)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, dims[0])
        
        # Encoder blocks (downsampling in terms of information compression)
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            self.encoder_blocks.append(
                UNetBlock(dims[i], dims[i + 1], dropout=dropout)
            )
        
        # Bottleneck
        bottleneck_dim = dims[depth]
        self.bottleneck = UNetBlock(bottleneck_dim, bottleneck_dim, dropout=dropout)
        
        # Decoder blocks (upsampling with skip connections)
        self.decoder_blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        for i in range(depth):
            # Skip connection projection (encoder dim -> decoder input)
            skip_in_dim = dims[depth - i]  # From encoder
            decoder_in_dim = dims[depth - i]  # From previous decoder
            combined_dim = skip_in_dim + decoder_in_dim
            
            self.skip_projections.append(
                nn.Linear(combined_dim, dims[depth - i - 1])
            )
            self.decoder_blocks.append(
                UNetBlock(dims[depth - i - 1], dims[depth - i - 1], dropout=dropout)
            )
        
        # Output projection
        self.output_proj = nn.Linear(dims[0], output_dim)
        
        # Initialize weights
        self._initialize_weights(init_scale)
    
    def _compute_dimensions(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        hidden_mult: float
    ) -> List[int]:
        """Compute dimensions for each level of the U-Net."""
        dims = []
        
        # Start from output_dim
        current_dim = output_dim
        dims.append(current_dim)
        
        # Encoder dimensions (increasing to bottleneck)
        for i in range(depth):
            current_dim = int(current_dim * hidden_mult)
            dims.append(current_dim)
        
        return dims
    
    def _initialize_weights(self, init_scale: float):
        """Initialize all weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net style MLP.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        
        Returns:
            pre_activations: Tensor of shape [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Encoder pass (save activations for skip connections)
        encoder_outputs = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder pass with skip connections
        for i, (decoder_block, skip_proj) in enumerate(
            zip(self.decoder_blocks, self.skip_projections)
        ):
            # Get corresponding encoder output for skip connection
            skip_output = encoder_outputs[-(i + 1)]
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_output], dim=-1)
            
            # Project and apply decoder block
            x = skip_proj(x)
            x = decoder_block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class SimpleSynapseModel(nn.Module):
    """
    Simplified synapse model for faster training/inference.
    
    A standard MLP with residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_scale: float = 0.02
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or output_dim * 2
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._initialize_weights(init_scale)
    
    def _initialize_weights(self, init_scale: float):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer in self.hidden_layers:
            x = x + layer(x)  # Residual connection
        
        return self.output_proj(x)
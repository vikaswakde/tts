"""
Lightweight encoder for the TTS system.
Converts mel spectrograms into compact representations efficiently.
Key features:
- CPU optimized
- Memory efficient
- Fast processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LightweightEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 80,  # Mel spectrogram channels
                 hidden_dim: int = 128, # Hidden layer size
                 num_layers: int = 3,   # Number of encoding layers
                 dropout: float = 0.1): # Dropout rate
        """
        Initialize the lightweight encoder.
        
        Args:
            input_dim: Number of input features (mel bands)
            hidden_dim: Size of hidden layers
            num_layers: Number of encoding layers
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dim)
            
        Returns:
            Encoded representation of shape (batch_size, time_steps, hidden_dim)
        """
        # Initial projection
        x = self.input_projection(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Final normalization
        x = self.norm(x)
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        """
        Single encoder layer with self-attention and feed-forward network.
        
        Args:
            hidden_dim: Size of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Simplified self-attention
        self.self_attention = SimplifiedAttention(hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder layer."""
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class SimplifiedAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        """
        Simplified self-attention mechanism optimized for CPU.
        
        Args:
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-attention."""
        # Generate query, key, value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores (scaled dot-product attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        return torch.matmul(attention, V) 
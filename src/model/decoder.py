"""
Lightweight decoder for the TTS system.
Converts encoded representations back into mel spectrograms.
Includes visualization capabilities to understand the generation process.

Think of it as teaching our model to "speak" from understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, Union

class LightweightDecoder(nn.Module):
    def __init__(self,
                 encoder_dim: int = 128,    # Dimension from encoder
                 hidden_dim: int = 256,     # Decoder hidden size
                 output_dim: int = 80,      # Mel spectrogram channels
                 num_layers: int = 4,       # More layers for generation
                 dropout: float = 0.1):
        """
        Initialize the decoder - think of this as the "speaking" part of our model.
        
        Args:
            encoder_dim: Size of encoded representations
            hidden_dim: Internal processing size
            output_dim: Size of mel spectrogram to generate
            num_layers: Number of processing layers
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers  # Store num_layers as instance variable
        
        # Project encoder output to decoder dimension
        self.input_projection = nn.Linear(encoder_dim, hidden_dim)
        
        # Decoder layers with visualization support
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Convert to mel spectrogram
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def forward(self, 
                encoder_output: torch.Tensor,
                store_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Generate mel spectrograms from encoder output.
        
        Args:
            encoder_output: Output from encoder (batch_size, time_steps, encoder_dim)
            store_attention: Whether to store attention weights for visualization
            
        Returns:
            mel_spec: Generated mel spectrogram
            attention_weights: Dictionary of attention weights if store_attention=True
        """
        # Clear previous attention weights
        if store_attention:
            self.attention_weights = []
        
        # Initial projection
        x = self.input_projection(encoder_output)
        
        # Apply decoder layers
        for i, layer in enumerate(self.decoder_layers):
            x, attention = layer(x, encoder_output)
            if store_attention:
                self.attention_weights.append(attention)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to mel spectrogram
        mel_spec = self.output_projection(x)
        
        # Return attention weights if requested
        if store_attention:
            return mel_spec, {"attention": self.attention_weights}
        return mel_spec, None
    
    def visualize_attention(self, save_path: Optional[str] = None):
        """
        Visualize the attention patterns of the decoder.
        
        Args:
            save_path: Path to save the visualization. If None, displays instead.
        """
        if not self.attention_weights:
            raise ValueError("No attention weights stored. Run forward with store_attention=True first.")
        
        num_layers = len(self.attention_weights)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2*num_layers))
        if num_layers == 1:
            axes = [axes]
            
        for i, attention in enumerate(self.attention_weights):
            # Get attention weights from first batch
            att_weights = attention[0].detach().cpu().numpy()
            
            # Plot attention heatmap
            im = axes[i].imshow(att_weights, aspect='auto', cmap='viridis')
            axes[i].set_title(f'Decoder Layer {i+1} Attention')
            plt.colorbar(im, ax=axes[i])
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        """
        Single decoder layer with cross-attention to encoder output.
        """
        super().__init__()
        
        # Self-attention for processing current state
        self.self_attention = SimplifiedAttention(hidden_dim)
        
        # Cross-attention to look at encoder output
        self.cross_attention = SimplifiedAttention(hidden_dim)
        
        # Processing network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention visualization support."""
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = residual + x
        
        # Cross-attention to encoder output
        residual = x
        x = self.norm2(x)
        x, attention_weights = self.cross_attention(x, return_attention=True)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, attention_weights

class SimplifiedAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        """
        Simplified attention mechanism optimized for CPU.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, 
                encoder_output: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute attention with optional visualization support.
        
        Args:
            x: Input tensor
            encoder_output: Optional encoder output for cross-attention
            return_attention: Whether to return attention weights
            
        Returns:
            If return_attention is True:
                tuple of (output tensor, attention weights)
            Otherwise:
                output tensor only
        """
        # Use encoder output for cross-attention if provided
        key_value_input = encoder_output if encoder_output is not None else x
        
        # Generate query, key, value
        Q = self.query(x)
        K = self.key(key_value_input)
        V = self.value(key_value_input)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        if return_attention:
            return output, attention_weights
        return output 
"""
Test cases for the LightweightDecoder.
"""
import pytest
import torch
import numpy as np
import os
from src.model.decoder import LightweightDecoder

@pytest.fixture
def decoder():
    """Create a LightweightDecoder instance for testing."""
    return LightweightDecoder()

def test_decoder_output_shape():
    """Test if decoder produces correct output shape."""
    decoder = LightweightDecoder(encoder_dim=128, hidden_dim=256, output_dim=80)
    batch_size = 2
    time_steps = 50
    
    # Create dummy encoder output
    encoder_output = torch.randn(batch_size, time_steps, 128)
    
    # Get decoder output
    mel_spec, _ = decoder(encoder_output)
    
    # Check output shape
    assert mel_spec.shape == (batch_size, time_steps, 80)

def test_attention_visualization(tmp_path):
    """Test attention visualization capability."""
    decoder = LightweightDecoder()
    batch_size = 2
    time_steps = 10
    
    # Create dummy encoder output
    encoder_output = torch.randn(batch_size, time_steps, 128)
    
    # Forward pass with attention storage
    mel_spec, attention_dict = decoder(encoder_output, store_attention=True)
    
    # Check if attention weights were stored
    assert attention_dict is not None
    assert "attention" in attention_dict
    assert len(attention_dict["attention"]) == decoder.num_layers
    
    # Test visualization saving
    vis_path = tmp_path / "attention_vis.png"
    decoder.visualize_attention(str(vis_path))
    assert vis_path.exists()

def test_decoder_small_input():
    """Test decoder with minimal input."""
    decoder = LightweightDecoder()
    
    # Single sample, single time step
    encoder_output = torch.randn(1, 1, 128)
    mel_spec, _ = decoder(encoder_output)
    
    assert mel_spec.shape == (1, 1, 80)

def test_decoder_parameters():
    """Test if decoder parameters are learnable."""
    decoder = LightweightDecoder()
    
    # Check if model has parameters
    param_count = sum(p.numel() for p in decoder.parameters())
    assert param_count > 0
    
    # Check if parameters require gradients
    has_grad = any(p.requires_grad for p in decoder.parameters())
    assert has_grad

def test_decoder_forward_pass():
    """Test complete forward pass through decoder."""
    decoder = LightweightDecoder()
    
    # Create input
    encoder_output = torch.randn(2, 10, 128)
    
    # Forward pass with attention
    mel_spec, attention_dict = decoder(encoder_output, store_attention=True)
    
    # Basic checks
    assert not torch.isnan(mel_spec).any()  # No NaN values
    assert mel_spec.requires_grad  # Output should require gradients
    assert attention_dict is not None  # Should have attention weights 
"""
Test cases for the LightweightEncoder.
"""
import pytest
import torch
import numpy as np
from src.model.encoder import LightweightEncoder

@pytest.fixture
def encoder():
    """Create a LightweightEncoder instance for testing."""
    return LightweightEncoder()

def test_encoder_output_shape():
    """Test if encoder produces correct output shape."""
    encoder = LightweightEncoder(input_dim=80, hidden_dim=128)
    batch_size = 2
    time_steps = 50
    
    # Create dummy input (batch_size, time_steps, input_dim)
    x = torch.randn(batch_size, time_steps, 80)
    
    # Get encoder output
    output = encoder(x)
    
    # Check output shape
    assert output.shape == (batch_size, time_steps, 128)

def test_encoder_small_input():
    """Test encoder with minimal input."""
    encoder = LightweightEncoder(input_dim=80, hidden_dim=128)
    
    # Single sample, single time step
    x = torch.randn(1, 1, 80)
    output = encoder(x)
    
    assert output.shape == (1, 1, 128)

def test_encoder_parameters():
    """Test if encoder parameters are learnable."""
    encoder = LightweightEncoder()
    
    # Check if model has parameters
    param_count = sum(p.numel() for p in encoder.parameters())
    assert param_count > 0
    
    # Check if parameters require gradients
    has_grad = any(p.requires_grad for p in encoder.parameters())
    assert has_grad

def test_encoder_forward_pass():
    """Test complete forward pass through encoder."""
    encoder = LightweightEncoder()
    
    # Create input
    x = torch.randn(2, 10, 80)  # (batch_size, time_steps, input_dim)
    
    # Forward pass
    output = encoder(x)
    
    # Basic checks
    assert not torch.isnan(output).any()  # No NaN values
    assert output.requires_grad  # Output should require gradients 
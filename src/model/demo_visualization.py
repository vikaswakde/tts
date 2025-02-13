"""
Demo script to visualize the decoder's attention patterns.
Shows how the model focuses on different parts of the input when generating speech.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.model.encoder import LightweightEncoder
from src.model.decoder import LightweightDecoder

def create_sample_input(duration_seconds: float = 0.5, sample_rate: int = 16000):
    """Create a sample input (sine wave) for demonstration."""
    # Create time points (using shorter duration and lower sample rate)
    t = torch.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    
    # Create a simple sine wave (reduced complexity)
    frequency = 440.0  # A4 note
    audio = torch.sin(2 * torch.pi * frequency * t)
    
    # Normalize
    audio = audio / audio.abs().max()
    return audio

def visualize_attention_demo():
    """
    Demonstrate the attention visualization capabilities.
    Shows how the decoder pays attention to different parts of the input.
    """
    print("Creating models...")
    # Create models with smaller dimensions
    encoder = LightweightEncoder(
        input_dim=40,      # Match mel spectrogram dimensions
        hidden_dim=64      # Reduced from 128
    )
    decoder = LightweightDecoder(
        encoder_dim=64,     # Match encoder hidden_dim
        hidden_dim=128,     # Reduced from 256
        num_layers=2        # Reduced from 4
    )
    
    print("Creating sample input...")
    # Create sample input
    audio = create_sample_input()
    
    # Use smaller sequence length
    max_length = 100  # Limit sequence length
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    # Add batch dimension and channel dimension
    audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
    
    print("Creating mel spectrogram...")
    # Create smaller mel spectrogram
    mel_spec = torch.randn(1, 40, audio.size(-1))  # Reduced from 80 to 40 mel bands
    
    # Transpose for encoder (batch, time, features)
    mel_spec = mel_spec.transpose(1, 2)
    
    # Convert to float32 for efficiency
    mel_spec = mel_spec.float()
    
    print("Step 1: Encoding mel spectrogram...")
    # Encode
    with torch.no_grad():  # No need for gradients in demo
        encoded = encoder(mel_spec)
        
        print("Step 2: Decoding with attention visualization...")
        # Decode with attention visualization
        mel_output, attention_dict = decoder(encoded, store_attention=True)
    
    print("Step 3: Creating visualization...")
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Save attention visualization
    decoder.visualize_attention(str(output_dir / "attention_patterns.png"))
    
    print(f"\nVisualization saved to {output_dir}/attention_patterns.png")
    print("\nWhat you're seeing:")
    print("- Each row shows one decoder layer's attention pattern")
    print("- Brighter colors mean stronger attention")
    print("- X-axis: Input sequence positions")
    print("- Y-axis: Output sequence positions")
    print("\nInterpretation:")
    print("- Look for diagonal patterns (local attention)")
    print("- Vertical stripes show global attention")
    print("- Scattered patterns show complex relationships")
    
    # Clean up
    del encoder, decoder, audio, mel_spec, encoded, mel_output, attention_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    try:
        visualize_attention_demo()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("If you're seeing memory errors, try reducing max_length or model dimensions further.")
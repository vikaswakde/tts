"""
Test cases for the training script.
Tests model setup, data setup, and configuration handling.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from src.training.train import setup_models, setup_data
from src.utils.config import MEL_CHANNELS, SAMPLE_RATE

@dataclass
class MockAudioInfo:
    """Mock audio info for testing."""
    sample_rate: int = SAMPLE_RATE
    num_frames: int = SAMPLE_RATE  # 1 second of audio
    num_channels: int = 1

@pytest.fixture
def mock_audio(monkeypatch):
    """Mock both torchaudio.info and torchaudio.load."""
    def mock_info(filepath):
        return MockAudioInfo()
    
    def mock_load(filepath):
        # Return dummy waveform and sample rate
        return torch.randn(1, SAMPLE_RATE), SAMPLE_RATE
    
    monkeypatch.setattr("torchaudio.info", mock_info)
    monkeypatch.setattr("torchaudio.load", mock_load)

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock LJSpeech dataset structure."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create wavs directory
    wavs_dir = temp_path / "wavs"
    wavs_dir.mkdir(parents=True)
    
    # Create metadata.csv with sample data
    metadata = pd.DataFrame({
        "file_id": ["LJ001-0001", "LJ001-0002"],
        "raw_text": ["Sample text one.", "Sample text two."],
        "normalized_text": ["Sample text one.", "Sample text two."]
    })
    
    # Save metadata with the correct format (pipe-separated)
    with open(temp_path / "metadata.csv", "w") as f:
        for _, row in metadata.iterrows():
            f.write(f"{row['file_id']}|{row['raw_text']}|{row['normalized_text']}\n")
    
    # Create dummy wav files
    for file_id in ["LJ001-0001", "LJ001-0002"]:
        # Create an empty file (torchaudio.info and load will be mocked)
        (wavs_dir / f"{file_id}.wav").touch()
    
    yield temp_path
    shutil.rmtree(temp_dir)

def test_model_setup():
    """Test model initialization."""
    encoder, decoder = setup_models(encoder_dim=64, hidden_dim=128)
    
    # Check encoder
    assert encoder.input_dim == MEL_CHANNELS
    assert encoder.hidden_dim == 64
    
    # Check decoder
    assert decoder.encoder_dim == 64
    assert decoder.hidden_dim == 128
    assert decoder.output_dim == MEL_CHANNELS

def test_model_output_shapes():
    """Test if model outputs have correct shapes."""
    encoder, decoder = setup_models(encoder_dim=64, hidden_dim=128)
    
    # Create dummy input
    batch_size = 2
    time_steps = 50
    mel_input = torch.randn(batch_size, time_steps, MEL_CHANNELS)
    
    # Test encoder output shape
    encoded = encoder(mel_input)
    assert encoded.shape == (batch_size, time_steps, 64)
    
    # Test decoder output shape
    mel_output, _ = decoder(encoded)
    assert mel_output.shape == (batch_size, time_steps, MEL_CHANNELS)

def test_data_setup(monkeypatch, temp_data_dir, mock_audio):
    """Test data loader setup with real dataset structure."""
    # Test data setup
    train_loader, val_loader = setup_data(
        data_dir=temp_data_dir,
        batch_size=1,  # Small batch size for testing
        val_split=0.5  # Equal split for testing
    )
    
    # Check data split (we have 2 samples total)
    train_dataset: Dataset = train_loader.dataset
    val_dataset: Dataset = val_loader.dataset
    assert len(train_dataset) == 1  # 50% of 2
    assert len(val_dataset) == 1    # 50% of 2
    
    # Check batch contents
    batch = next(iter(train_loader))
    assert "text_padded" in batch
    assert "mel_padded" in batch
    assert "text_lengths" in batch
    assert "mel_lengths" in batch
    assert batch["batch_size"] == 1

def test_model_parameter_counts():
    """Test if models have reasonable parameter counts."""
    encoder, decoder = setup_models(encoder_dim=64, hidden_dim=128)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    # Check if parameter counts are reasonable
    assert 1000 < encoder_params < 1000000  # Adjust these bounds as needed
    assert 1000 < decoder_params < 1000000  # Adjust these bounds as needed
    
    # Check if parameters are trainable
    assert all(p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for p in decoder.parameters()) 
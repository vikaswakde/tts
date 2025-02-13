"""
Test cases for the MelSpectrogramGenerator class.
"""
import pytest
import numpy as np
import librosa
import warnings

# Filter out the specific librosa deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message="path is deprecated. Use files()")

from src.audio_synthesis.mel_generator import MelSpectrogramGenerator

@pytest.fixture
def mel_gen():
    """Create a MelSpectrogramGenerator instance for testing."""
    return MelSpectrogramGenerator()

def test_empty_input(mel_gen):
    """Test handling of empty input."""
    empty_audio = np.array([])
    with pytest.raises(ValueError, match="Input audio cannot be empty"):
        mel_gen.generate(empty_audio)

def test_short_input(mel_gen):
    """Test handling of input shorter than n_fft."""
    short_audio = np.zeros(1024)  # Less than n_fft (2048)
    with pytest.raises(ValueError, match="Input audio length .* must be at least n_fft"):
        mel_gen.generate(short_audio)

def test_basic_generation(mel_gen):
    """Test basic mel spectrogram generation."""
    # Generate a simple sine wave
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(22050 * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    mel_spec = mel_gen.generate(audio)
    
    # Check output shape and type
    assert isinstance(mel_spec, np.ndarray)
    assert len(mel_spec.shape) == 2
    assert mel_spec.shape[0] == mel_gen.n_mels

def test_visualization(mel_gen, tmp_path):
    """Test visualization saving."""
    # Generate test audio
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(22050 * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    # Generate mel spectrogram
    mel_spec = mel_gen.generate(audio)
    
    # Save visualization
    output_file = tmp_path / "test_mel_spec.png"
    mel_gen.save_visualization(mel_spec, str(output_file))
    
    # Check if file was created
    assert output_file.exists() 
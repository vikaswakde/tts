"""
MelSpectrogramGenerator for TTS system.
Converts phonemes to mel spectrograms which represent audio features.
"""
import numpy as np
import librosa
from typing import List, Optional

class MelSpectrogramGenerator:
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: Optional[float] = 8000.0):
        """
        Initialize the Mel Spectrogram Generator.
        
        Args:
            sample_rate: Audio sample rate (default: 22050 Hz)
            n_fft: FFT window size
            hop_length: Number of samples between windows
            win_length: Window size
            n_mels: Number of mel bands
            mel_fmin: Minimum frequency
            mel_fmax: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        # Create mel filter bank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )

    def generate(self, audio_wave: np.ndarray) -> np.ndarray:
        """
        Generate mel spectrogram from audio waveform.
        
        Args:
            audio_wave: Audio waveform as numpy array
            
        Returns:
            Mel spectrogram as numpy array
            
        Raises:
            ValueError: If input audio is empty or invalid
        """
        # Check for empty input
        if len(audio_wave) == 0:
            raise ValueError("Input audio cannot be empty")

        # Ensure audio is mono
        if len(audio_wave.shape) > 1:
            audio_wave = audio_wave.mean(axis=1)

        # Validate input length
        if len(audio_wave) < self.n_fft:
            raise ValueError(f"Input audio length ({len(audio_wave)}) must be at least n_fft ({self.n_fft})")

        # Step 1: Compute power spectrogram
        D = librosa.stft(audio_wave,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length)
        power_spec = np.abs(D) ** 2

        # Step 2: Convert to mel scale
        mel_spec = np.dot(self.mel_basis, power_spec)

        # Step 3: Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec

    def visualize(self, mel_spec: np.ndarray, title: str = "Mel Spectrogram") -> None:
        """
        Visualize the mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram to visualize
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec,
                               y_axis='mel',
                               x_axis='time',
                               sr=self.sample_rate,
                               fmin=self.mel_fmin,
                               fmax=self.mel_fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def save_visualization(self, mel_spec: np.ndarray, 
                         filename: str, 
                         title: str = "Mel Spectrogram") -> None:
        """
        Save the mel spectrogram visualization to a file.
        
        Args:
            mel_spec: Mel spectrogram to visualize
            filename: Output file path
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec,
                               y_axis='mel',
                               x_axis='time',
                               sr=self.sample_rate,
                               fmin=self.mel_fmin,
                               fmax=self.mel_fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close() 
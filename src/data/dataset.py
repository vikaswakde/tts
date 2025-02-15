"""
LJSpeech dataset handler for TTS system.
Processes audio files and corresponding text from LJSpeech dataset.
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, Tuple, List
import pandas as pd
from pathlib import Path

from ..text_processing.normalizer import TextNormalizer
from ..text_processing.phoneme_converter import PhonemeConverter
from ..audio_synthesis.mel_generator import MelSpectrogramGenerator
from ..utils.config import SAMPLE_RATE, MEL_CHANNELS

class LJSpeechDataset(Dataset):
    """Dataset class for loading and preprocessing LJSpeech data."""
    
    def __init__(self, 
                 root_dir: str,
                 max_audio_length: int = 10,  # Maximum audio length in seconds
                 max_text_length: int = 200): # Maximum text length in characters
        """
        Initialize the dataset.
        
        Args:
            root_dir: Path to LJSpeech dataset
            max_audio_length: Maximum allowed audio length in seconds
            max_text_length: Maximum allowed text length
        """
        self.root_dir = Path(root_dir)
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
        # Initialize processors
        self.text_normalizer = TextNormalizer()
        self.phoneme_converter = PhonemeConverter()
        self.mel_generator = MelSpectrogramGenerator(
            sample_rate=SAMPLE_RATE,
            n_mels=MEL_CHANNELS
        )
        
        # Load metadata
        metadata_path = self.root_dir / "metadata.csv"
        self.metadata = pd.read_csv(
            metadata_path,
            sep="|",
            names=["file_id", "raw_text", "normalized_text"],
            quoting=3  # QUOTE_NONE
        )
        
        # Filter long samples to avoid memory issues
        self._filter_long_samples()
        
    def _filter_long_samples(self):
        """Remove samples that exceed maximum lengths."""
        # Filter by text length
        text_mask = self.metadata["normalized_text"].str.len() <= self.max_text_length
        
        # Filter by audio length (checking actual files)
        audio_mask = []
        for idx in range(len(self.metadata)):
            try:
                info = torchaudio.info(self._get_audio_path(idx))
                duration = info.num_frames / info.sample_rate
                audio_mask.append(duration <= self.max_audio_length)
            except Exception:
                audio_mask.append(False)
        
        # Apply both filters
        self.metadata = self.metadata[text_mask & pd.Series(audio_mask)]
        self.metadata = self.metadata.reset_index(drop=True)
        
    def _get_audio_path(self, idx: int) -> Path:
        """Get path to audio file."""
        file_id = self.metadata.iloc[idx]["file_id"]
        return self.root_dir / "wavs" / f"{file_id}.wav"
        
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from dataset.
        
        Returns dictionary containing:
            - text: normalized and converted to phonemes
            - mel: mel spectrogram of audio
            - audio: raw audio waveform
            - text_length: length of text
            - mel_length: length of mel spectrogram
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Process text
        text = self.text_normalizer.normalize_text(row["normalized_text"])
        phonemes = self.phoneme_converter.convert_to_phonemes(text)
        
        # Load and process audio
        audio_path = self._get_audio_path(idx)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                sample_rate, SAMPLE_RATE
            )(waveform)
        
        # Generate mel spectrogram
        mel_spec = self.mel_generator.generate(waveform.numpy().squeeze())
        
        return {
            "text": torch.tensor([ord(c) for c in phonemes], dtype=torch.long),
            "mel": torch.FloatTensor(mel_spec),
            "audio": waveform.squeeze(),
            "text_length": torch.tensor(len(phonemes)),
            "mel_length": torch.tensor(mel_spec.shape[1])
        } 
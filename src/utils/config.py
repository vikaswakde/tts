"""
Configuration settings for the TTS system.
Includes paths, model parameters, and audio processing settings.
"""
from pathlib import Path
import os

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Audio settings
SAMPLE_RATE = 22050  # Standard sampling rate
MEL_CHANNELS = 80    # Number of mel channels
HOP_LENGTH = 256    # Number of frames between windows
WIN_LENGTH = 1024   # Each frame length
N_FFT = 2048       # Length of FFT window

# Model settings
BATCH_SIZE = 8     # Small batch size for CPU
LEARNING_RATE = 1e-4
MAX_TEXT_LENGTH = 200  # Maximum text length to process

# Training settings
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 
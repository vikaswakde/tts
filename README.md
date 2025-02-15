\*\*\*\*# Zero-Cost TTS System

A lightweight, CPU-optimized Text-to-Speech system designed to run on limited hardware.

## System Requirements

- Python 3.8+
- CPU: AMD Ryzen 3 3200U
- RAM: 9.6GB
- Storage: ~342GB **available**

## Project Structure

```
tts/
├── src/                  # Source code
│   ├── text_processing/  # Text normalization and phoneme conversion
│   ├── audio_synthesis/  # Voice generation components
│   └── utils/           # Helper functions and utilities
├── data/                # Dataset storage
│   ├── raw/            # Original audio files
│   └── processed/      # Preprocessed data
└── tests/              # Unit tests
```

## Setup Instructions

1. Create and activate virtual environment:

```bash
python -m venv tts-env
source tts-env/bin/activate  # Linux/Mac
```

2. Install dependencies:

```bash
pip install torch torchaudio numpy scipy librosa phonemizer
pip install pandas tqdm matplotlib
```

## Features

- Lightweight transformer-based architecture
- CPU-optimized processing
- Real-time inference capability
- Support for English (expandable to other languages)

## Development Status

- [x] Project setup
- [x] Configuration
- [x] Text normalization
- [x] Audio processing
- [ ] Model implementation
- [ ] Training pipeline
- [ ] Inference API

## License

Apache 2.0

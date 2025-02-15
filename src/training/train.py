"""
Main training script for TTS system.
Handles dataset setup, model initialization, and training execution.
"""

import torch
from pathlib import Path
import argparse
import logging
from typing import Tuple

from ..model.encoder import LightweightEncoder
from ..model.decoder import LightweightDecoder
from ..data.dataset import LJSpeechDataset
from ..data.dataloader import create_dataloader
from ..utils.config import (
    BATCH_SIZE, LEARNING_RATE, DATA_DIR,
    CHECKPOINT_DIR, MEL_CHANNELS
)
from .trainer import TTSTrainer

def setup_models(encoder_dim: int = 128,
                hidden_dim: int = 256) -> Tuple[LightweightEncoder, LightweightDecoder]:
    """
    Initialize encoder and decoder models.
    
    Args:
        encoder_dim: Dimension of encoder output
        hidden_dim: Hidden dimension for models
        
    Returns:
        Tuple of (encoder, decoder)
    """
    # Create encoder
    encoder = LightweightEncoder(
        input_dim=MEL_CHANNELS,
        hidden_dim=encoder_dim
    )
    
    # Create decoder
    decoder = LightweightDecoder(
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
        output_dim=MEL_CHANNELS
    )
    
    return encoder, decoder

def setup_data(data_dir: Path,
               batch_size: int,
               val_split: float = 0.1) -> Tuple[torch.utils.data.DataLoader,
                                              torch.utils.data.DataLoader]:
    """
    Set up training and validation data loaders.
    
    Args:
        data_dir: Directory containing dataset
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = LJSpeechDataset(str(data_dir))
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train TTS model")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                       help="Directory containing LJSpeech dataset")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                       help="Checkpoint saving frequency (epochs)")
    parser.add_argument("--resume_checkpoint", type=str,
                       help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create directories
        data_dir = Path(args.data_dir)
        checkpoint_dir = Path(CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        logger.info(f"Using device: {device}")
        
        # Setup models
        encoder, decoder = setup_models()
        encoder.to(device)
        decoder.to(device)
        
        # Setup trainer
        trainer = TTSTrainer(
            encoder=encoder,
            decoder=decoder,
            learning_rate=args.learning_rate,
            checkpoint_dir=checkpoint_dir
        )
        
        # Resume from checkpoint if specified
        if args.resume_checkpoint:
            checkpoint_path = Path(args.resume_checkpoint)
            if checkpoint_path.exists():
                trainer.load_checkpoint(checkpoint_path)
                logger.info(f"Resumed from checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        # Setup data
        train_loader, val_loader = setup_data(
            data_dir=data_dir,
            batch_size=args.batch_size
        )
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Start training
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            device=device,
            checkpoint_frequency=args.checkpoint_freq
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
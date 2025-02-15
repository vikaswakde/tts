"""
Main trainer module for TTS system.
Implements progressive learning and efficient CPU training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import logging
from tqdm import tqdm
import numpy as np

from ..model.encoder import LightweightEncoder
from ..model.decoder import LightweightDecoder
from ..utils.config import BATCH_SIZE, LEARNING_RATE, CHECKPOINT_DIR
from ..data.dataset import LJSpeechDataset
from ..data.dataloader import create_dataloader

class TTSTrainer:
    def __init__(self,
                 encoder: LightweightEncoder,
                 decoder: LightweightDecoder,
                 learning_rate: float = LEARNING_RATE,
                 checkpoint_dir: Path = CHECKPOINT_DIR):
        """
        Initialize the TTS trainer.
        
        Args:
            encoder: Encoder model instance
            decoder: Decoder model instance
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.encoder = encoder
        self.decoder = decoder
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize optimizers
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        # Initialize loss functions
        self.mel_loss = nn.MSELoss()  # For spectrogram reconstruction
        self.duration_loss = nn.L1Loss()  # For timing accuracy
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': 0
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for training progress."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, 
                    train_loader: torch.utils.data.DataLoader,
                    device: str = 'cpu') -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            device: Device to train on ('cpu' or 'cuda')
            
        Returns:
            Average loss for the epoch
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.history["epoch"]+1}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get batch data
                text_padded = batch["text_padded"].to(device)
                mel_padded = batch["mel_padded"].to(device)
                text_lengths = batch["text_lengths"].to(device)
                mel_lengths = batch["mel_lengths"].to(device)
                
                # Clear gradients
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                # Forward pass
                encoded = self.encoder(text_padded)
                mel_output, attention_dict = self.decoder(encoded, store_attention=True)
                
                # Calculate losses
                mel_loss = self.mel_loss(mel_output, mel_padded)
                
                # Calculate duration loss using attention
                duration_loss = self._calculate_duration_loss(
                    attention_dict["attention"][-1],  # Use last layer's attention
                    text_lengths,
                    mel_lengths
                )
                
                # Combine losses
                loss = mel_loss * 0.8 + duration_loss * 0.2
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                
                # Update weights
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Clean up to save memory
                del encoded, mel_output, attention_dict
                if batch_idx % 10 == 0:  # Every 10 batches
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / num_batches
    
    def _calculate_duration_loss(self,
                               attention_weights: torch.Tensor,
                               text_lengths: torch.Tensor,
                               mel_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate duration loss using attention weights.
        Encourages monotonic attention (proper timing).
        """
        # Calculate expected durations from attention
        durations = attention_weights.sum(dim=2)  # [batch_size, text_len]
        
        # Create target durations (uniform distribution as ideal case)
        target_durations = torch.zeros_like(durations)
        for i, (text_len, mel_len) in enumerate(zip(text_lengths, mel_lengths)):
            target_durations[i, :text_len] = mel_len / text_len
            
        return self.duration_loss(durations, target_durations)
    
    def validate(self,
                val_loader: torch.utils.data.DataLoader,
                device: str = 'cpu') -> float:
        """Run validation."""
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                text_padded = batch["text_padded"].to(device)
                mel_padded = batch["mel_padded"].to(device)
                text_lengths = batch["text_lengths"].to(device)
                mel_lengths = batch["mel_lengths"].to(device)
                
                # Forward pass through encoder with mel input
                encoded = self.encoder(mel_padded.transpose(1, 2))  # Transpose to match encoder input shape
                mel_output, _ = self.decoder(encoded)
                
                # Calculate loss
                loss = self.mel_loss(mel_output, mel_padded.transpose(1, 2))  # Transpose to match output shape
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        
        self.history = checkpoint['history']
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
    def train(self,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             num_epochs: int,
             device: str = 'cpu',
             checkpoint_frequency: int = 5):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            device: Device to train on
            checkpoint_frequency: How often to save checkpoints
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                
                # Training
                train_loss = self.train_epoch(train_loader, device)
                
                # Validation
                val_loss = self.validate(val_loader, device)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['epoch'] = epoch
                
                # Log progress
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Save checkpoint
                if (epoch + 1) % checkpoint_frequency == 0:
                    self.save_checkpoint(epoch + 1, val_loss)
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
        finally:
            # Final checkpoint
            self.save_checkpoint(self.history['epoch'] + 1, val_loss)
            
            # Training summary
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f}s") 
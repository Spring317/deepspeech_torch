#!/usr/bin/env python3
"""
Vietnamese ASR Finetuning Pipeline using DeepSpeech
This pipeline provides a complete workflow for finetuning DeepSpeech on Vietnamese audio data.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VietnameseASRPipeline:
    """Complete pipeline for Vietnamese ASR finetuning with DeepSpeech"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize vocabulary and character mappings
        self.setup_vocabulary()
        
    def setup_vocabulary(self):
        """Setup Vietnamese character vocabulary"""
        # Vietnamese characters including tones and common punctuation
        vietnamese_chars = [
            'a', 'á', 'à', 'ả', 'ã', 'ạ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
            'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'b', 'c', 'd', 'đ', 'e', 'é',
            'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'f', 'g',
            'h', 'i', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'j', 'k', 'l', 'm', 'n',
            'o', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ',
            'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'p', 'q', 'r', 's', 't', 'u',
            'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'v',
            'w', 'x', 'y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'z', ' ', "'", '-'
        ]
        
        # Add special tokens
        self.vocab = ['<blank>', '<unk>'] + vietnamese_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        text = text.lower().strip()
        sequence = []
        for char in text:
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(self.char_to_idx['<unk>'])
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text"""
        chars = []
        for idx in sequence:
            if idx < len(self.vocab) and idx != self.char_to_idx['<blank>']:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)


class VietnameseAudioDataset(Dataset):
    """Dataset class for Vietnamese audio data"""
    
    def __init__(self, data_list: List[Dict], pipeline: VietnameseASRPipeline, 
                 sample_rate: int = 16000):
        self.data_list = data_list
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(item['audio_path'])
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Convert text to sequence
        text_sequence = self.pipeline.text_to_sequence(item['transcript'])
        
        return {
            'waveform': waveform.squeeze(0),
            'text_sequence': torch.tensor(text_sequence, dtype=torch.long),
            'transcript': item['transcript'],
            'audio_path': item['audio_path']
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    waveforms = [item['waveform'] for item in batch]
    text_sequences = [item['text_sequence'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    
    # Pad waveforms
    waveform_lengths = torch.tensor([len(w) for w in waveforms])
    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    
    # Pad text sequences
    text_lengths = torch.tensor([len(t) for t in text_sequences])
    padded_texts = pad_sequence(text_sequences, batch_first=True, padding_value=0)
    
    return {
        'waveforms': padded_waveforms,
        'waveform_lengths': waveform_lengths,
        'text_sequences': padded_texts,
        'text_lengths': text_lengths,
        'transcripts': transcripts,
        'audio_paths': audio_paths
    }


class DeepSpeechModel(nn.Module):
    """PyTorch implementation of DeepSpeech model"""
    
    def __init__(self, vocab_size: int, n_features: int = 80, 
                 hidden_size: int = 512, num_layers: int = 5):
        super(DeepSpeechModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        self.bn2 = nn.BatchNorm2d(32)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size()
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_size = conv_output_size
        
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
            )
            input_size = hidden_size * 2  # bidirectional
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size * 2, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def _get_conv_output_size(self):
        """Calculate the output size after convolutions"""
        # This is an approximation - you might need to adjust based on your input size
        # Assuming input spectrograms have 161 frequency bins
        size = self.n_features
        # After conv1: (161 - 41 + 2*20) / 2 + 1 = 81
        size = (size - 41 + 2*20) // 2 + 1
        # After conv2: (81 - 21 + 2*10) / 1 + 1 = 81
        size = (size - 21 + 2*10) // 1 + 1
        return size * 32  # 32 channels from conv2
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Add channel dimension for conv layers
        x = x.unsqueeze(1)  # (batch, 1, time, freq)
        
        # Convolutional layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        
        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class ASRTrainer:
    """Trainer class for ASR model"""
    
    def __init__(self, model: nn.Module, pipeline: VietnameseASRPipeline, config: Dict):
        self.model = model
        self.pipeline = pipeline
        self.config = config
        self.device = pipeline.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            eps=1e-8  # Add epsilon for numerical stability
        )
        
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        
        # Learning rate scheduler with more aggressive reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=3
        )
    
    def calculate_output_lengths(self, input_lengths):
        """
        Calculate output sequence lengths after convolutions
        DeepSpeech model applies 2 conv layers with stride (2,2) and (2,1)
        """
        # After conv1: stride=(2,2) -> time dimension is divided by 2
        lengths = (input_lengths - 1) // 2 + 1
        # After conv2: stride=(2,1) -> time dimension is divided by 2 again
        lengths = (lengths - 1) // 2 + 1
        return lengths
    
    def compute_features(self, waveforms):
        """Compute mel-scale spectrograms"""
        # Move waveforms to device first
        waveforms = waveforms.to(self.device)
        
        # Convert to mel-scale spectrograms
        # Using n_fft=512 gives n_freqs=257, so we use n_mels=80 to avoid warning
        # n_mels=80 is a common choice that works well for speech recognition
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=80  # Reduced from 128 to 80 to avoid warning
        ).to(self.device)
        
        # Apply log transform
        features = torch.log(mel_specgram(waveforms) + 1e-8)
        
        # Transpose to (batch, time, freq)
        features = features.transpose(1, 2)
        
        return features
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            waveforms = batch['waveforms'].to(self.device)
            waveform_lengths = batch['waveform_lengths'].to(self.device)
            text_sequences = batch['text_sequences'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            
            # Compute features
            features = self.compute_features(waveforms)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(features, waveform_lengths)
            
            # Get actual output sequence length from model
            batch_size, actual_time_steps, vocab_size = logits.shape
            
            # Calculate output lengths after convolutions
            # We need to use the actual feature lengths, not waveform lengths
            feature_time_steps = features.shape[1]  # Time dimension of features
            output_lengths = self.calculate_output_lengths(torch.tensor([feature_time_steps] * batch_size))
            output_lengths = output_lengths.to(self.device)
            
            # Clamp output_lengths to not exceed actual output size
            output_lengths = torch.clamp(output_lengths, max=actual_time_steps)
            
            # Transpose for CTC: (time, batch, vocab)
            logits = logits.transpose(0, 1)
            
            # CTC Loss
            loss = self.criterion(
                logits, text_sequences, output_lengths, text_lengths
            )
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping batch due to invalid loss: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Move to device
                waveforms = batch['waveforms'].to(self.device)
                waveform_lengths = batch['waveform_lengths'].to(self.device)
                text_sequences = batch['text_sequences'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                
                # Compute features
                features = self.compute_features(waveforms)
                
                # Forward pass
                logits = self.model(features, waveform_lengths)
                
                # Get actual output sequence length from model
                batch_size, actual_time_steps, vocab_size = logits.shape
                
                # Calculate output lengths after convolutions
                feature_time_steps = features.shape[1]
                output_lengths = self.calculate_output_lengths(torch.tensor([feature_time_steps] * batch_size))
                output_lengths = output_lengths.to(self.device)
                
                # Clamp output_lengths to not exceed actual output size
                output_lengths = torch.clamp(output_lengths, max=actual_time_steps)
                
                # Transpose for CTC
                logits = logits.transpose(0, 1)
                
                # CTC Loss
                loss = self.criterion(
                    logits, text_sequences, output_lengths, text_lengths
                )
                
                # Skip invalid losses
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / num_batches
    
    def decode_predictions(self, logits):
        """Simple greedy decoding"""
        # Get most likely characters
        predicted_indices = torch.argmax(logits, dim=-1)
        
        decoded_texts = []
        for sequence in predicted_indices:
            # Remove blanks and consecutive duplicates
            prev_char = None
            chars = []
            for idx in sequence:
                char_idx = idx.item()
                if char_idx != 0 and char_idx != prev_char:  # 0 is blank
                    chars.append(char_idx)
                prev_char = char_idx
            
            decoded_text = self.pipeline.sequence_to_text(chars)
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
    def train(self, train_dataloader, val_dataloader, num_epochs: int):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate
            val_loss = self.validate(val_dataloader)
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if old_lr != new_lr:
                logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                logger.info("New best model saved!")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'vocab': self.pipeline.vocab,
            'char_to_idx': self.pipeline.char_to_idx,
            'config': self.config
        }
        
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        else:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')


def load_data_list(data_path: str) -> List[Dict]:
    """
    Load data list from JSON file.
    Expected format: [{"audio_path": "path/to/audio.wav", "transcript": "text"}, ...]
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list


def main():
    parser = argparse.ArgumentParser(description="Vietnamese ASR Finetuning with DeepSpeech")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data JSON')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize pipeline
    pipeline = VietnameseASRPipeline(config)
    
    # Load datasets
    train_data = load_data_list(args.train_data)
    val_data = load_data_list(args.val_data)
    
    train_dataset = VietnameseAudioDataset(train_data, pipeline)
    val_dataset = VietnameseAudioDataset(val_data, pipeline)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4)
    )
    
    # Initialize model
    model = DeepSpeechModel(
        vocab_size=pipeline.vocab_size,
        n_features=config.get('n_features', 161),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 5)
    )
    
    # Initialize trainer
    trainer = ASRTrainer(model, pipeline, config)
    
    # Start training
    logger.info("Starting training...")
    trainer.train(train_dataloader, val_dataloader, config['num_epochs'])
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
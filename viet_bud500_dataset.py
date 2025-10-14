#!/usr/bin/env python3
"""
Direct dataset class for viet_bud500 from Hugging Face
Works directly with the AudioEncoder datatype without saving to disk
"""

import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietBud500Dataset(Dataset):
    """
    Dataset class for viet_bud500 that works directly with Hugging Face datasets
    Handles AudioEncoder datatype from torchaudio
    """
    
    def __init__(self, split: str = 'train', pipeline=None, 
                 sample_rate: int = 16000, cache_dir: str = "/storage/student5/quydx/",
                 min_duration: float = 0.5, max_duration: float = 20.0):
        """
        Args:
            split: Dataset split ('train', 'validation', 'test')
            pipeline: VietnameseASRPipeline instance for text processing
            sample_rate: Target sample rate
            cache_dir: Cache directory for Hugging Face datasets
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
        """
        self.split = split
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        
        # Load dataset from Hugging Face
        logger.info(f"Loading viet_bud500 dataset (split: {split})...")
        try:
            self.dataset = load_dataset("linhtran92/viet_bud500", split=split, cache_dir=cache_dir)
            logger.info(f"Loaded {len(self.dataset)} samples from {split} split")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        # Filter samples by duration
        self.valid_indices = []
        logger.info("Filtering samples by duration...")
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                audio_data = sample.get('audio', None)
                
                if audio_data is None:
                    continue
                
                # Get duration
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    duration = len(audio_data['array']) / audio_data.get('sampling_rate', sample_rate)
                else:
                    # Try to process the sample to get duration
                    waveform = self._extract_audio(audio_data)
                    duration = waveform.shape[0] / sample_rate
                
                # Filter by duration
                if min_duration <= duration <= max_duration:
                    self.valid_indices.append(idx)
                    
            except Exception as e:
                logger.debug(f"Skipping sample {idx}: {e}")
                continue
        
        logger.info(f"Filtered to {len(self.valid_indices)} valid samples")
    
    def _extract_audio(self, audio_data):
        """Extract audio waveform from various formats"""
        if isinstance(audio_data, dict):
            # Audio is a dictionary with array and sampling_rate
            if 'array' in audio_data and 'sampling_rate' in audio_data:
                waveform = torch.tensor(audio_data['array'], dtype=torch.float32)
                sr = audio_data['sampling_rate']
                
                # Ensure waveform is 1D
                if waveform.dim() > 1:
                    waveform = waveform.squeeze()
                
            elif 'path' in audio_data:
                # Load from path
                waveform, sr = torchaudio.load(audio_data['path'])
                waveform = waveform.squeeze()
            else:
                raise ValueError(f"Unknown audio format: {audio_data.keys()}")
                
        elif isinstance(audio_data, torch.Tensor):
            waveform = audio_data.squeeze()
            sr = self.sample_rate
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze()
        
        # Normalize
        if waveform.abs().max() > 0:
            waveform = waveform / (waveform.abs().max() + 1e-8)
        
        return waveform
    
    def _extract_transcript(self, sample):
        """Extract transcript from sample"""
        # Try different possible field names
        transcript = sample.get('sentence', 
                               sample.get('transcript', 
                                         sample.get('text', '')))
        return transcript.strip() if transcript else ''
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual dataset index
        dataset_idx = self.valid_indices[idx]
        sample = self.dataset[dataset_idx]
        
        # Extract audio
        audio_data = sample['audio']
        waveform = self._extract_audio(audio_data)
        
        # Extract transcript
        transcript = self._extract_transcript(sample)
        
        # Convert text to sequence if pipeline is provided
        if self.pipeline:
            text_sequence = self.pipeline.text_to_sequence(transcript)
        else:
            text_sequence = []
        
        return {
            'waveform': waveform,
            'text_sequence': torch.tensor(text_sequence, dtype=torch.long),
            'transcript': transcript,
            'audio_path': f"viet_bud500_{self.split}_{dataset_idx}"
        }


def get_viet_bud500_dataloaders(pipeline, batch_size: int = 8, 
                                num_workers: int = 4, cache_dir: str =  "/storage/student5/quydx/",
                                min_duration: float = 0.5, max_duration: float = 20.0):
    """
    Get dataloaders for viet_bud500 dataset
    
    Args:
        pipeline: VietnameseASRPipeline instance
        batch_size: Batch size
        num_workers: Number of worker processes
        cache_dir: Cache directory for Hugging Face datasets
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
    
    Returns:
        train_dataloader, val_dataloader, test_dataloader (if available)
    """
    from torch.utils.data import DataLoader
    from pipeline_vietnamese_asr import collate_fn
    
    # Load dataset info to check available splits
    from datasets import get_dataset_config_names, get_dataset_split_names
    
    try:
        available_splits = get_dataset_split_names("linhtran92/viet_bud500")
        logger.info(f"Available splits: {available_splits}")
    except:
        # Default splits
        available_splits = ['train', 'validation', 'test']
        logger.info(f"Using default splits: {available_splits}")
    
    dataloaders = {}
    
    # Create datasets for available splits
    if 'train' in available_splits:
        logger.info("Creating train dataset...")
        train_dataset = VietBud500Dataset(
            split='train', 
            pipeline=pipeline, 
            cache_dir=cache_dir,
            min_duration=min_duration,
            max_duration=max_duration
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders['train'] = train_dataloader
    
    if 'validation' in available_splits:
        logger.info("Creating validation dataset...")
        val_dataset = VietBud500Dataset(
            split='validation',
            pipeline=pipeline,
            cache_dir=cache_dir,
            min_duration=min_duration,
            max_duration=max_duration
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders['validation'] = val_dataloader
    
    if 'test' in available_splits:
        logger.info("Creating test dataset...")
        test_dataset = VietBud500Dataset(
            split='test',
            pipeline=pipeline,
            cache_dir=cache_dir,
            min_duration=min_duration,
            max_duration=max_duration
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders['test'] = test_dataloader
    
    return dataloaders


if __name__ == "__main__":
    # Test the dataset loading
    import argparse
    from pipeline_vietnamese_asr import VietnameseASRPipeline
    
    parser = argparse.ArgumentParser(description="Test viet_bud500 dataset loading")
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    
    # Initialize pipeline
    config = {'sample_rate': 16000}
    pipeline = VietnameseASRPipeline(config)
    
    # Create dataset
    dataset = VietBud500Dataset(
        split=args.split,
        pipeline=pipeline,
        cache_dir=args.cache_dir
    )
    
    logger.info(f"\nDataset created successfully!")
    logger.info(f"Total samples: {len(dataset)}")
    
    # Test first sample
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"\nFirst sample:")
        logger.info(f"  Waveform shape: {sample['waveform'].shape}")
        logger.info(f"  Text sequence length: {len(sample['text_sequence'])}")
        logger.info(f"  Transcript: {sample['transcript']}")

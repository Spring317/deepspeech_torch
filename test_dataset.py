#!/usr/bin/env python3
"""
Test script to verify viet_bud500 dataset loading
"""

import torch
from viet_bud500_dataset import VietBud500Dataset, get_viet_bud500_dataloaders
from pipeline_vietnamese_asr import VietnameseASRPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test loading the viet_bud500 dataset"""
    
    logger.info("="*50)
    logger.info("Testing viet_bud500 Dataset Loading")
    logger.info("="*50)
    
    # Initialize pipeline
    config = {
        'sample_rate': 16000,
        'batch_size': 4,
        'num_workers': 2
    }
    
    pipeline = VietnameseASRPipeline(config)
    
    # Test loading train split
    logger.info("\n1. Testing train split loading...")
    try:
        train_dataset = VietBud500Dataset(
            split='train',
            pipeline=pipeline,
            min_duration=0.5,
            max_duration=20.0
        )
        logger.info(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        
        # Test getting a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"  Sample waveform shape: {sample['waveform'].shape}")
            logger.info(f"  Sample transcript: {sample['transcript'][:100]}...")
            logger.info(f"  Text sequence length: {len(sample['text_sequence'])}")
    except Exception as e:
        logger.error(f"✗ Error loading train dataset: {e}")
    
    # Test loading validation split
    logger.info("\n2. Testing validation split loading...")
    try:
        val_dataset = VietBud500Dataset(
            split='validation',
            pipeline=pipeline,
            min_duration=0.5,
            max_duration=20.0
        )
        logger.info(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
        
        if len(val_dataset) > 0:
            sample = val_dataset[0]
            logger.info(f"  Sample waveform shape: {sample['waveform'].shape}")
            logger.info(f"  Sample transcript: {sample['transcript'][:100]}...")
    except Exception as e:
        logger.error(f"✗ Error loading validation dataset: {e}")
    
    # Test dataloader creation
    logger.info("\n3. Testing dataloader creation...")
    try:
        dataloaders = get_viet_bud500_dataloaders(
            pipeline=pipeline,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        logger.info(f"✓ Created {len(dataloaders)} dataloaders")
        
        for split_name, dataloader in dataloaders.items():
            logger.info(f"  {split_name}: {len(dataloader)} batches, {len(dataloader.dataset)} samples")
        
        # Test getting a batch
        if 'train' in dataloaders:
            logger.info("\n4. Testing batch retrieval...")
            train_dataloader = dataloaders['train']
            batch = next(iter(train_dataloader))
            
            logger.info(f"✓ Batch retrieved successfully")
            logger.info(f"  Waveforms shape: {batch['waveforms'].shape}")
            logger.info(f"  Waveform lengths: {batch['waveform_lengths']}")
            logger.info(f"  Text sequences shape: {batch['text_sequences'].shape}")
            logger.info(f"  Text lengths: {batch['text_lengths']}")
            logger.info(f"  Number of transcripts: {len(batch['transcripts'])}")
            logger.info(f"  Sample transcript: {batch['transcripts'][0][:100]}...")
            
    except Exception as e:
        logger.error(f"✗ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*50)
    logger.info("Dataset loading test completed!")
    logger.info("="*50)


if __name__ == "__main__":
    test_dataset_loading()

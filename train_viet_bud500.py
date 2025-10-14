#!/usr/bin/env python3
"""
Training script for Vietnamese ASR using viet_bud500 dataset from Hugging Face
"""

import os
import torch
import argparse
import json
import logging
from pathlib import Path

from pipeline_vietnamese_asr import (
    VietnameseASRPipeline, 
    DeepSpeechModel, 
    ASRTrainer
)
from viet_bud500_dataset import get_viet_bud500_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Vietnamese ASR with viet_bud500 dataset"
    )
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config JSON file')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for Hugging Face datasets')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--min_duration', type=float, default=0.5,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=20.0,
                       help='Maximum audio duration in seconds')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    if args.cache_dir:
        config['cache_dir'] = args.cache_dir
    
    # Initialize pipeline
    logger.info("Initializing Vietnamese ASR pipeline...")
    pipeline = VietnameseASRPipeline(config)
    
    # Load dataloaders for viet_bud500
    logger.info("Loading viet_bud500 dataset from Hugging Face...")
    dataloaders = get_viet_bud500_dataloaders(
        pipeline=pipeline,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        cache_dir=config.get('cache_dir', None),
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    if 'train' not in dataloaders:
        logger.error("Training data not available!")
        return
    
    if 'validation' not in dataloaders:
        logger.warning("Validation data not available. Using train data for validation.")
        dataloaders['validation'] = dataloaders['train']
    
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['validation']
    
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Initialize model
    logger.info("Initializing DeepSpeech model...")
    model = DeepSpeechModel(
        vocab_size=pipeline.vocab_size,
        n_features=config.get('n_features', 161),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 5)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = ASRTrainer(model, pipeline, config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    config_save_path = checkpoint_dir / 'training_config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {config_save_path}")
    
    # Start training
    logger.info("="*50)
    logger.info("Starting training with viet_bud500 dataset")
    logger.info("="*50)
    
    num_epochs = config.get('num_epochs', 50)
    
    # If resuming, adjust num_epochs
    if start_epoch > 0:
        remaining_epochs = num_epochs - start_epoch
        logger.info(f"Remaining epochs: {remaining_epochs}")
        trainer.train(train_dataloader, val_dataloader, remaining_epochs)
    else:
        trainer.train(train_dataloader, val_dataloader, num_epochs)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

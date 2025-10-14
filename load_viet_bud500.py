#!/usr/bin/env python3
"""
Data loader for linhtran92/viet_bud500 dataset from Hugging Face
This dataset uses AudioEncoder datatype from torchaudio
"""

import os
import json
import torch
import torchaudio
from datasets import load_dataset
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietBud500Loader:
    """Loader for viet_bud500 dataset"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        
    def load_dataset(self, split: str = None):
        """Load the viet_bud500 dataset from Hugging Face"""
        logger.info(f"Loading viet_bud500 dataset from Hugging Face...")
        
        try:
            # Load dataset
            if split:
                dataset = load_dataset("linhtran92/viet_bud500", split=split, cache_dir=self.cache_dir)
            else:
                dataset = load_dataset("linhtran92/viet_bud500", cache_dir=self.cache_dir)
            
            logger.info(f"Dataset loaded successfully!")
            if isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name}: {len(split_data)} samples")
            else:
                logger.info(f"  Total samples: {len(dataset)}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def process_sample(self, sample: Dict, target_sr: int = 16000):
        """
        Process a single sample from the dataset
        Handle AudioEncoder datatype from torchaudio
        """
        # Extract audio
        # The dataset might have audio as a dictionary with 'array', 'sampling_rate', 'path'
        # or directly as encoded audio
        audio_data = sample.get('audio', None)
        
        if audio_data is None:
            raise ValueError("No audio field found in sample")
        
        # Handle different audio formats
        if isinstance(audio_data, dict):
            # Audio is a dictionary with array and sampling_rate
            if 'array' in audio_data and 'sampling_rate' in audio_data:
                waveform = torch.tensor(audio_data['array'], dtype=torch.float32)
                sr = audio_data['sampling_rate']
                
                # Ensure waveform is 1D or 2D
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # Add channel dimension
                
            elif 'path' in audio_data:
                # Load from path
                waveform, sr = torchaudio.load(audio_data['path'])
            else:
                raise ValueError(f"Unknown audio format in sample: {audio_data.keys()}")
        else:
            # Audio might be a tensor or bytes
            if isinstance(audio_data, torch.Tensor):
                waveform = audio_data
                sr = target_sr  # Assume target sample rate
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
        
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Calculate duration
        duration = waveform.shape[1] / target_sr
        
        # Extract transcript
        # The dataset might have 'sentence', 'transcript', 'text' or similar field
        transcript = sample.get('sentence', 
                               sample.get('transcript', 
                                         sample.get('text', '')))
        
        return {
            'waveform': waveform.squeeze(0),
            'transcript': transcript.strip() if transcript else '',
            'duration': duration,
            'sample_id': sample.get('id', sample.get('idx', '')),
        }
    
    def create_manifest_from_dataset(self, dataset, output_file: str, 
                                     min_duration: float = 0.5,
                                     max_duration: float = 20.0,
                                     save_audio: bool = False,
                                     audio_output_dir: str = None):
        """
        Create manifest file from the dataset
        Optionally save audio files to disk
        """
        manifest = []
        
        if save_audio and audio_output_dir:
            audio_dir = Path(audio_output_dir)
            audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(dataset)} samples...")
        
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Process sample
                processed = self.process_sample(sample)
                
                # Filter by duration
                if processed['duration'] < min_duration or processed['duration'] > max_duration:
                    continue
                
                # Save audio if requested
                if save_audio and audio_output_dir:
                    audio_filename = f"audio_{idx:06d}.wav"
                    audio_path = audio_dir / audio_filename
                    
                    # Save audio file
                    torchaudio.save(
                        str(audio_path),
                        processed['waveform'].unsqueeze(0),
                        16000
                    )
                    
                    manifest_item = {
                        'audio_path': str(audio_path),
                        'transcript': processed['transcript'],
                        'duration': processed['duration'],
                        'sample_id': processed.get('sample_id', f'sample_{idx}')
                    }
                else:
                    # Store reference to dataset sample
                    manifest_item = {
                        'dataset_index': idx,
                        'transcript': processed['transcript'],
                        'duration': processed['duration'],
                        'sample_id': processed.get('sample_id', f'sample_{idx}')
                    }
                
                manifest.append(manifest_item)
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        # Save manifest
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created manifest with {len(manifest)} samples")
        logger.info(f"Total duration: {sum(item['duration'] for item in manifest)/3600:.2f} hours")
        
        return manifest
    
    def analyze_dataset(self, dataset):
        """Analyze dataset statistics"""
        durations = []
        transcript_lengths = []
        
        logger.info("Analyzing dataset...")
        
        for idx, sample in enumerate(tqdm(dataset, desc="Analyzing")):
            try:
                processed = self.process_sample(sample)
                durations.append(processed['duration'])
                transcript_lengths.append(len(processed['transcript']))
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(durations)}")
        print(f"Total duration: {sum(durations)/3600:.2f} hours")
        print(f"Average duration: {sum(durations)/len(durations):.2f} seconds")
        print(f"Min duration: {min(durations):.2f} seconds")
        print(f"Max duration: {max(durations):.2f} seconds")
        print(f"Average transcript length: {sum(transcript_lengths)/len(transcript_lengths):.1f} characters")
        
        # Duration distribution
        import numpy as np
        percentiles = [25, 50, 75, 90, 95, 99]
        duration_percentiles = np.percentile(durations, percentiles)
        
        print(f"\nDuration Percentiles:")
        for p, d in zip(percentiles, duration_percentiles):
            print(f"  {p}th percentile: {d:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Load viet_bud500 dataset from Hugging Face")
    parser.add_argument('--output_dir', type=str, default='viet_bud500_data',
                       help='Output directory for manifests and audio files')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for Hugging Face datasets')
    parser.add_argument('--save_audio', action='store_true',
                       help='Save audio files to disk')
    parser.add_argument('--min_duration', type=float, default=0.5,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=20.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset statistics')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loader
    loader = VietBud500Loader(cache_dir=args.cache_dir)
    
    # Load dataset
    dataset = loader.load_dataset()
    
    # Check if dataset has splits
    if isinstance(dataset, dict):
        # Process each split
        for split_name, split_data in dataset.items():
            logger.info(f"\nProcessing {split_name} split...")
            
            if args.analyze:
                loader.analyze_dataset(split_data)
            
            # Create manifest
            manifest_file = output_dir / f"{split_name}_manifest.json"
            audio_dir = output_dir / "audio" / split_name if args.save_audio else None
            
            loader.create_manifest_from_dataset(
                split_data,
                str(manifest_file),
                args.min_duration,
                args.max_duration,
                args.save_audio,
                audio_dir
            )
            
            logger.info(f"Saved {split_name} manifest to {manifest_file}")
    else:
        # Single split dataset
        if args.analyze:
            loader.analyze_dataset(dataset)
        
        # Create manifest
        manifest_file = output_dir / "manifest.json"
        audio_dir = output_dir / "audio" if args.save_audio else None
        
        loader.create_manifest_from_dataset(
            dataset,
            str(manifest_file),
            args.min_duration,
            args.max_duration,
            args.save_audio,
            audio_dir
        )
        
        logger.info(f"Saved manifest to {manifest_file}")
    
    logger.info("\nDataset loading completed!")
    logger.info(f"Manifests saved to: {output_dir}")


if __name__ == "__main__":
    main()

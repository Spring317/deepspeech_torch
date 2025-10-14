#!/usr/bin/env python3
"""
Data preprocessing utilities for Vietnamese ASR
This script helps prepare Vietnamese audio data for training.
"""

import os
import json
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseTextPreprocessor:
    """Preprocessor for Vietnamese text normalization"""
    
    def __init__(self):
        # Common Vietnamese text normalizations
        self.normalizations = {
            # Numbers to words (basic examples)
            r'\b0\b': 'không',
            r'\b1\b': 'một',
            r'\b2\b': 'hai',
            r'\b3\b': 'ba',
            r'\b4\b': 'bốn',
            r'\b5\b': 'năm',
            r'\b6\b': 'sáu',
            r'\b7\b': 'bảy',
            r'\b8\b': 'tám',
            r'\b9\b': 'chín',
            r'\b10\b': 'mười',
            
            # Common abbreviations
            r'\bdr\.\b': 'bác sĩ',
            r'\bmr\.\b': 'ông',
            r'\bmrs\.\b': 'bà',
            r'\bms\.\b': 'cô',
            r'\bvs\.\b': 'và',
            r'\betc\.\b': 'vân vân',
            
            # Clean up multiple spaces
            r'\s+': ' ',
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text"""
        text = text.lower().strip()
        
        # Apply normalizations
        for pattern, replacement in self.normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove unwanted characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\'-]', '', text)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class AudioPreprocessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def process_audio(self, audio_path: str, output_path: str = None) -> Tuple[str, float]:
        """Process audio file and return path and duration"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Calculate duration
            duration = len(audio) / self.target_sr
            
            if output_path:
                # Save processed audio
                sf.write(output_path, audio, self.target_sr)
                return output_path, duration
            else:
                return audio_path, duration
                
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None, 0.0


def create_data_manifest(data_dir: str, output_file: str, min_duration: float = 0.5, 
                        max_duration: float = 20.0, preprocess_audio: bool = False):
    """
    Create data manifest from directory structure.
    Expected structure:
    data_dir/
        audio/
            file1.wav
            file2.wav
        transcripts/
            file1.txt
            file2.txt
    """
    
    audio_dir = Path(data_dir) / "audio"
    transcript_dir = Path(data_dir) / "transcripts"
    
    if not audio_dir.exists() or not transcript_dir.exists():
        raise ValueError(f"Expected 'audio' and 'transcripts' subdirectories in {data_dir}")
    
    text_processor = VietnameseTextPreprocessor()
    audio_processor = AudioPreprocessor() if preprocess_audio else None
    
    manifest = []
    
    # Get all audio files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flac"))
    
    for audio_file in tqdm(audio_files, desc="Processing files"):
        # Find corresponding transcript
        transcript_file = transcript_dir / f"{audio_file.stem}.txt"
        
        if not transcript_file.exists():
            logger.warning(f"No transcript found for {audio_file.name}")
            continue
        
        # Read transcript
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading transcript {transcript_file}: {e}")
            continue
        
        # Normalize text
        normalized_transcript = text_processor.normalize_text(transcript)
        
        if not normalized_transcript:
            logger.warning(f"Empty transcript after normalization: {transcript_file}")
            continue
        
        # Process audio if requested
        if preprocess_audio and audio_processor:
            processed_dir = Path(data_dir) / "processed_audio"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / audio_file.name
            
            audio_path, duration = audio_processor.process_audio(
                str(audio_file), str(processed_path)
            )
        else:
            # Just get duration
            try:
                audio, sr = librosa.load(str(audio_file), sr=None)
                duration = len(audio) / sr
                audio_path = str(audio_file)
            except Exception as e:
                logger.error(f"Error loading audio {audio_file}: {e}")
                continue
        
        # Filter by duration
        if duration < min_duration or duration > max_duration:
            logger.debug(f"Skipping {audio_file.name}: duration {duration:.2f}s outside range")
            continue
        
        # Add to manifest
        manifest.append({
            "audio_path": audio_path,
            "transcript": normalized_transcript,
            "duration": duration,
            "original_transcript": transcript
        })
    
    # Save manifest
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created manifest with {len(manifest)} samples")
    logger.info(f"Total duration: {sum(item['duration'] for item in manifest):.2f} hours")
    
    return manifest


def split_data(manifest_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split data into train/validation/test sets"""
    
    with open(manifest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle data
    import random
    random.shuffle(data)
    
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    base_path = Path(manifest_file).parent
    
    with open(base_path / "train_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(base_path / "val_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    if test_data:
        with open(base_path / "test_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")


def analyze_dataset(manifest_file: str):
    """Analyze dataset statistics"""
    
    with open(manifest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    durations = [item['duration'] for item in data]
    transcript_lengths = [len(item['transcript']) for item in data]
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Total duration: {sum(durations):.2f} hours")
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
    parser = argparse.ArgumentParser(description="Vietnamese ASR Data Preprocessing")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing audio and transcripts subdirectories')
    parser.add_argument('--output', type=str, default='manifest.json',
                       help='Output manifest file')
    parser.add_argument('--min_duration', type=float, default=0.5,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=20.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--preprocess_audio', action='store_true',
                       help='Apply audio preprocessing')
    parser.add_argument('--split', action='store_true',
                       help='Split data into train/val/test sets')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset statistics')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Create manifest
    logger.info("Creating data manifest...")
    create_data_manifest(
        args.data_dir, 
        args.output,
        args.min_duration,
        args.max_duration,
        args.preprocess_audio
    )
    
    # Analyze dataset
    if args.analyze:
        analyze_dataset(args.output)
    
    # Split data
    if args.split:
        logger.info("Splitting data...")
        split_data(args.output, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
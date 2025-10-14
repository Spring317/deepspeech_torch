#!/usr/bin/env python3
"""
Inference script for Vietnamese ASR using trained DeepSpeech model
"""

import torch
import torchaudio
import json
import argparse
from pathlib import Path
import logging
from pipeline_vietnamese_asr import DeepSpeechModel, VietnameseASRPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRInference:
    """Inference class for Vietnamese ASR"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize pipeline with vocab from checkpoint
        config = checkpoint['config']
        self.pipeline = VietnameseASRPipeline(config)
        self.pipeline.vocab = checkpoint['vocab']
        self.pipeline.char_to_idx = checkpoint['char_to_idx']
        self.pipeline.idx_to_char = {idx: char for char, idx in checkpoint['char_to_idx'].items()}
        self.pipeline.vocab_size = len(self.pipeline.vocab)
        
        # Initialize model
        self.model = DeepSpeechModel(
            vocab_size=self.pipeline.vocab_size,
            n_features=config.get('n_features', 161),
            hidden_size=config.get('hidden_size', 512),
            num_layers=config.get('num_layers', 5)
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature extractor
        self.mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=161
        ).to(self.device)
        
        logger.info("Model loaded successfully")
    
    def preprocess_audio(self, audio_path: str):
        """Preprocess audio for inference"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        return waveform.squeeze(0)
    
    def compute_features(self, waveform):
        """Compute mel-scale spectrograms"""
        waveform = waveform.to(self.device)
        
        # Add batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute mel spectrogram
        features = torch.log(self.mel_specgram(waveform) + 1e-8)
        
        # Transpose to (batch, time, freq)
        features = features.transpose(1, 2)
        
        return features
    
    def decode_greedy(self, logits):
        """Greedy decoding"""
        # Get most likely characters
        predicted_indices = torch.argmax(logits, dim=-1)
        
        # Remove blanks and consecutive duplicates
        sequence = predicted_indices.squeeze(0).cpu().numpy()
        
        prev_char = None
        chars = []
        for idx in sequence:
            char_idx = int(idx)
            if char_idx != 0 and char_idx != prev_char:  # 0 is blank
                chars.append(char_idx)
            prev_char = char_idx
        
        decoded_text = self.pipeline.sequence_to_text(chars)
        return decoded_text
    
    def decode_beam_search(self, logits, beam_width: int = 5):
        """Beam search decoding (simplified version)"""
        # This is a simplified beam search - you might want to use a more sophisticated implementation
        # For now, we'll use greedy decoding
        return self.decode_greedy(logits)
    
    def transcribe(self, audio_path: str, decode_method: str = 'greedy', beam_width: int = 5):
        """Transcribe audio file"""
        logger.info(f"Transcribing {audio_path}")
        
        # Preprocess audio
        waveform = self.preprocess_audio(audio_path)
        
        # Compute features
        features = self.compute_features(waveform)
        
        # Get waveform length
        waveform_length = torch.tensor([len(waveform)])
        
        # Inference
        with torch.no_grad():
            logits = self.model(features, waveform_length)
        
        # Decode
        if decode_method == 'greedy':
            transcript = self.decode_greedy(logits)
        elif decode_method == 'beam_search':
            transcript = self.decode_beam_search(logits, beam_width)
        else:
            raise ValueError(f"Unknown decode method: {decode_method}")
        
        return transcript.strip()
    
    def transcribe_batch(self, audio_paths: list, decode_method: str = 'greedy', beam_width: int = 5):
        """Transcribe multiple audio files"""
        results = []
        
        for audio_path in audio_paths:
            try:
                transcript = self.transcribe(audio_path, decode_method, beam_width)
                results.append({
                    'audio_path': audio_path,
                    'transcript': transcript,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'transcript': '',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Vietnamese ASR Inference with DeepSpeech")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio', type=str,
                       help='Path to audio file for single inference')
    parser.add_argument('--audio_list', type=str,
                       help='Path to text file containing list of audio files')
    parser.add_argument('--output', type=str,
                       help='Path to output JSON file (for batch processing)')
    parser.add_argument('--decode_method', type=str, default='greedy',
                       choices=['greedy', 'beam_search'],
                       help='Decoding method')
    parser.add_argument('--beam_width', type=int, default=5,
                       help='Beam width for beam search decoding')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize inference
    asr = ASRInference(args.checkpoint, args.device)
    
    if args.audio:
        # Single file inference
        transcript = asr.transcribe(args.audio, args.decode_method, args.beam_width)
        print(f"Transcript: {transcript}")
        
    elif args.audio_list:
        # Batch inference
        with open(args.audio_list, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Transcribing {len(audio_paths)} files...")
        results = asr.transcribe_batch(audio_paths, args.decode_method, args.beam_width)
        
        # Print results
        for result in results:
            if result['status'] == 'success':
                print(f"{result['audio_path']}: {result['transcript']}")
            else:
                print(f"{result['audio_path']}: ERROR - {result['error']}")
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    else:
        parser.error("Either --audio or --audio_list must be provided")


if __name__ == "__main__":
    main()
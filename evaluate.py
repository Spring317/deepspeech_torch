#!/usr/bin/env python3
"""
Evaluation script for Vietnamese ASR model
Calculate WER, CER, and other metrics
"""

import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict
import editdistance
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluator for ASR models"""
    
    def __init__(self):
        pass
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for evaluation"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = self.normalize_text(reference).split()
        hyp_words = self.normalize_text(hypothesis).split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        
        edit_distance = editdistance.eval(ref_words, hyp_words)
        wer = edit_distance / len(ref_words)
        
        return wer
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        ref_chars = list(self.normalize_text(reference).replace(' ', ''))
        hyp_chars = list(self.normalize_text(hypothesis).replace(' ', ''))
        
        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0
        
        edit_distance = editdistance.eval(ref_chars, hyp_chars)
        cer = edit_distance / len(ref_chars)
        
        return cer
    
    def calculate_accuracy(self, reference: str, hypothesis: str) -> float:
        """Calculate sentence-level accuracy"""
        ref_normalized = self.normalize_text(reference)
        hyp_normalized = self.normalize_text(hypothesis)
        
        return 1.0 if ref_normalized == hyp_normalized else 0.0
    
    def evaluate_dataset(self, results: List[Dict]) -> Dict:
        """Evaluate entire dataset"""
        total_wer = 0.0
        total_cer = 0.0
        total_accuracy = 0.0
        valid_samples = 0
        
        wer_scores = []
        cer_scores = []
        accuracy_scores = []
        
        for result in results:
            if result.get('status') != 'success':
                continue
            
            reference = result.get('reference', result.get('transcript', ''))
            hypothesis = result.get('hypothesis', result.get('predicted_transcript', ''))
            
            if not reference or not hypothesis:
                continue
            
            wer = self.calculate_wer(reference, hypothesis)
            cer = self.calculate_cer(reference, hypothesis)
            accuracy = self.calculate_accuracy(reference, hypothesis)
            
            wer_scores.append(wer)
            cer_scores.append(cer)
            accuracy_scores.append(accuracy)
            
            total_wer += wer
            total_cer += cer
            total_accuracy += accuracy
            valid_samples += 1
        
        if valid_samples == 0:
            return {
                'wer': 0.0,
                'cer': 0.0,
                'accuracy': 0.0,
                'total_samples': 0,
                'valid_samples': 0
            }
        
        avg_wer = total_wer / valid_samples
        avg_cer = total_cer / valid_samples
        avg_accuracy = total_accuracy / valid_samples
        
        return {
            'wer': avg_wer,
            'cer': avg_cer,
            'accuracy': avg_accuracy,
            'total_samples': len(results),
            'valid_samples': valid_samples,
            'wer_scores': wer_scores,
            'cer_scores': cer_scores,
            'accuracy_scores': accuracy_scores
        }
    
    def print_detailed_results(self, evaluation_results: Dict):
        """Print detailed evaluation results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Total samples: {evaluation_results['total_samples']}")
        print(f"Valid samples: {evaluation_results['valid_samples']}")
        print(f"Success rate: {evaluation_results['valid_samples']/evaluation_results['total_samples']*100:.2f}%")
        
        print(f"\nWord Error Rate (WER): {evaluation_results['wer']*100:.2f}%")
        print(f"Character Error Rate (CER): {evaluation_results['cer']*100:.2f}%")
        print(f"Sentence Accuracy: {evaluation_results['accuracy']*100:.2f}%")
        
        if 'wer_scores' in evaluation_results:
            wer_scores = evaluation_results['wer_scores']
            cer_scores = evaluation_results['cer_scores']
            
            print(f"\nWER Statistics:")
            print(f"  Min: {min(wer_scores)*100:.2f}%")
            print(f"  Max: {max(wer_scores)*100:.2f}%")
            print(f"  Median: {sorted(wer_scores)[len(wer_scores)//2]*100:.2f}%")
            
            print(f"\nCER Statistics:")
            print(f"  Min: {min(cer_scores)*100:.2f}%")
            print(f"  Max: {max(cer_scores)*100:.2f}%")
            print(f"  Median: {sorted(cer_scores)[len(cer_scores)//2]*100:.2f}%")
        
        print("="*50)


def load_ground_truth(gt_file: str) -> Dict[str, str]:
    """Load ground truth transcripts"""
    ground_truth = {}
    
    if gt_file.endswith('.json'):
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            audio_path = item.get('audio_path', '')
            transcript = item.get('transcript', '')
            if audio_path and transcript:
                # Use filename as key
                key = Path(audio_path).name
                ground_truth[key] = transcript
    
    else:
        # Assume text file with format: filename.wav|transcript
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    filename, transcript = line.split('|', 1)
                    ground_truth[filename.strip()] = transcript.strip()
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese ASR Model")
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--ground_truth', type=str,
                       help='Path to ground truth file (JSON or text)')
    parser.add_argument('--output', type=str,
                       help='Path to save detailed evaluation results')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed per-sample results')
    
    args = parser.parse_args()
    
    # Load predictions
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Load ground truth if provided
    ground_truth = {}
    if args.ground_truth:
        ground_truth = load_ground_truth(args.ground_truth)
    
    # Match predictions with ground truth
    matched_results = []
    for pred in predictions:
        audio_path = pred.get('audio_path', '')
        hypothesis = pred.get('transcript', '')
        
        # Find ground truth
        filename = Path(audio_path).name
        reference = ground_truth.get(filename, '')
        
        if reference:
            matched_results.append({
                'audio_path': audio_path,
                'reference': reference,
                'hypothesis': hypothesis,
                'status': pred.get('status', 'success')
            })
        else:
            # If no ground truth file provided, assume predictions contain reference
            reference = pred.get('reference', pred.get('ground_truth', ''))
            if reference:
                matched_results.append({
                    'audio_path': audio_path,
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'status': pred.get('status', 'success')
                })
    
    if not matched_results:
        logger.error("No matching samples found between predictions and ground truth")
        return
    
    # Evaluate
    evaluator = ASREvaluator()
    evaluation_results = evaluator.evaluate_dataset(matched_results)
    
    # Print results
    evaluator.print_detailed_results(evaluation_results)
    
    # Print detailed per-sample results if requested
    if args.detailed:
        print("\nDETAILED PER-SAMPLE RESULTS:")
        print("-"*80)
        
        for i, result in enumerate(matched_results):
            if result['status'] != 'success':
                continue
            
            wer = evaluator.calculate_wer(result['reference'], result['hypothesis'])
            cer = evaluator.calculate_cer(result['reference'], result['hypothesis'])
            
            print(f"\nSample {i+1}:")
            print(f"Audio: {Path(result['audio_path']).name}")
            print(f"Reference: {result['reference']}")
            print(f"Hypothesis: {result['hypothesis']}")
            print(f"WER: {wer*100:.2f}%, CER: {cer*100:.2f}%")
    
    # Save results if output path provided
    if args.output:
        output_data = {
            'evaluation_summary': evaluation_results,
            'detailed_results': matched_results
        }
        
        # Remove score lists to reduce file size
        if 'wer_scores' in output_data['evaluation_summary']:
            del output_data['evaluation_summary']['wer_scores']
            del output_data['evaluation_summary']['cer_scores']
            del output_data['evaluation_summary']['accuracy_scores']
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
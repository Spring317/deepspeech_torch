#!/bin/bash

# Vietnamese ASR Training Pipeline
# This script runs the complete pipeline from data preparation to evaluation

set -e  # Exit on error

echo "Vietnamese ASR Finetuning Pipeline"
echo "=================================="

# Configuration
DATA_DIR="data"
MANIFEST_FILE="manifest.json"
CONFIG_FILE="config.json"
CHECKPOINT_DIR="checkpoints"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found!"
    echo "Please organize your data as described in README.md"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo "Step 1: Data Preprocessing"
echo "========================="
python data_preprocessing.py \
    --data_dir "$DATA_DIR" \
    --output "$MANIFEST_FILE" \
    --min_duration 0.5 \
    --max_duration 20.0 \
    --preprocess_audio \
    --split \
    --analyze

if [ $? -ne 0 ]; then
    echo "Error: Data preprocessing failed!"
    exit 1
fi

echo ""
echo "Step 2: Training"
echo "==============="
python pipeline_vietnamese_asr.py \
    --config "$CONFIG_FILE" \
    --train_data train_manifest.json \
    --val_data val_manifest.json

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "Step 3: Inference on Test Set"
echo "============================="
if [ -f "test_manifest.json" ]; then
    # Extract audio paths from test manifest
    python -c "
import json
with open('test_manifest.json', 'r') as f:
    data = json.load(f)
with open('test_audio_list.txt', 'w') as f:
    for item in data:
        f.write(item['audio_path'] + '\\n')
"
    
    python inference.py \
        --checkpoint "$CHECKPOINT_DIR/best_model.pth" \
        --audio_list test_audio_list.txt \
        --output test_predictions.json \
        --decode_method greedy
    
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed!"
        exit 1
    fi
    
    echo ""
    echo "Step 4: Evaluation"
    echo "=================="
    python evaluate.py \
        --predictions test_predictions.json \
        --ground_truth test_manifest.json \
        --output evaluation_results.json \
        --detailed
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed!"
        exit 1
    fi
    
    echo ""
    echo "Pipeline completed successfully!"
    echo "==============================="
    echo "Results saved to:"
    echo "- Model checkpoint: $CHECKPOINT_DIR/best_model.pth"
    echo "- Test predictions: test_predictions.json"
    echo "- Evaluation results: evaluation_results.json"
    
else
    echo "No test set found. Skipping evaluation."
    echo "Training completed successfully!"
    echo "Model checkpoint saved to: $CHECKPOINT_DIR/best_model.pth"
fi
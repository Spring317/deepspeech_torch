#!/bin/bash

# Quick start script for training Vietnamese ASR with vlsp2020_vinai_100h dataset

set -e  # Exit on error

echo "================================================"
echo "Vietnamese ASR Training with vlsp2020_vinai_100h"
echo "================================================"
echo ""

# Check if config exists
if [ ! -f "config.json" ]; then
    echo "Error: config.json not found!"
    exit 1
fi

# Step 1: Test dataset loading
echo "Step 1: Testing dataset loading..."
echo "=================================="
python test_dataset.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Dataset loading test failed!"
    echo "Please check your internet connection and ensure the datasets library is installed."
    exit 1
fi

echo ""
echo "✓ Dataset loading test passed!"
echo ""

# Step 2: Ask user if they want to continue with training
read -p "Do you want to start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Create checkpoints directory
mkdir -p checkpoints

# Step 3: Start training
echo ""
echo "Step 2: Starting training..."
echo "============================="
echo "Training logs will be displayed below."
echo "Press Ctrl+C to stop training."
echo ""

python train_vlsp2020.py --config config.json

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "Training completed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Check your model in: checkpoints/best_model.pth"
echo "2. Run inference: python inference.py --checkpoint checkpoints/best_model.pth --audio your_audio.wav"
echo "3. Evaluate on test set: python evaluate.py --predictions predictions.json --ground_truth test_manifest.json"
echo ""

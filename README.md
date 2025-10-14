# Vietnamese ASR Finetuning Pipeline with DeepSpeech

This project provides a complete pipeline for finetuning DeepSpeech on Vietnamese speech recognition tasks using PyTorch.

## Features

- **Complete Vietnamese ASR Pipeline**: End-to-end training pipeline optimized for Vietnamese language
- **DeepSpeech Architecture**: PyTorch implementation of DeepSpeech with convolutional and LSTM layers
- **Vietnamese Text Processing**: Specialized text normalization for Vietnamese characters and tones
- **Audio Preprocessing**: Comprehensive audio processing utilities
- **Evaluation Metrics**: WER, CER, and accuracy calculation
- **Inference Tools**: Easy-to-use inference scripts for transcription

## Requirements

```bash
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.0
soundfile>=0.10.0
transformers>=4.0.0
editdistance>=0.5.0
numpy>=1.19.0
tqdm>=4.60.0
pathlib
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install torch torchaudio librosa soundfile transformers editdistance numpy tqdm
```

## Data Preparation

### Data Structure

Organize your Vietnamese audio data in the following structure:

```
your_data/
├── audio/
│   ├── file001.wav
│   ├── file002.wav
│   └── ...
└── transcripts/
    ├── file001.txt
    ├── file002.txt
    └── ...
```

### Data Preprocessing

Use the data preprocessing script to prepare your data:

```bash
python data_preprocessing.py \
    --data_dir /path/to/your_data \
    --output manifest.json \
    --min_duration 0.5 \
    --max_duration 20.0 \
    --preprocess_audio \
    --split \
    --analyze
```

This will:
- Create a manifest file with audio paths and transcripts
- Normalize Vietnamese text
- Filter audio by duration
- Split data into train/validation/test sets
- Analyze dataset statistics

## Training

### Configuration

Edit `config.json` to adjust training parameters:

```json
{
  "learning_rate": 1e-4,
  "batch_size": 8,
  "num_epochs": 50,
  "weight_decay": 1e-5,
  "n_features": 161,
  "hidden_size": 512,
  "num_layers": 5,
  "save_interval": 5,
  "num_workers": 4,
  "checkpoint_dir": "checkpoints",
  "sample_rate": 16000
}
```

### Start Training

```bash
python pipeline_vietnamese_asr.py \
    --config config.json \
    --train_data train_manifest.json \
    --val_data val_manifest.json
``` 
The training script will:
- Initialize the DeepSpeech model with Vietnamese vocabulary
- Train using CTC loss
- Save checkpoints regularly
- Apply learning rate scheduling
- Validate on validation set

## Inference

### Single File Transcription

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --audio path/to/audio.wav \
    --decode_method greedy
```

### Batch Transcription

Create a text file with audio file paths:

```
audio1.wav
audio2.wav
audio3.wav
```

Then run:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --audio_list audio_files.txt \
    --output results.json \
    --decode_method beam_search \
    --beam_width 5
```

## Evaluation

Evaluate your model using WER, CER, and accuracy metrics:

```bash
python evaluate.py \
    --predictions results.json \
    --ground_truth test_manifest.json \
    --output evaluation_results.json \
    --detailed
```

## Model Architecture

The DeepSpeech model consists of:

1. **Convolutional Layers**: Two conv2d layers for feature extraction
2. **Bidirectional LSTM**: Multiple LSTM layers for sequence modeling
3. **Output Layer**: Linear layer mapping to Vietnamese character vocabulary
4. **CTC Loss**: Connectionist Temporal Classification for alignment-free training

### Vietnamese Vocabulary

The model uses a comprehensive Vietnamese character set including:
- Base characters: a, b, c, d, đ, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
- Vowels with tones: á, à, ả, ã, ạ, ă, ắ, ằ, ẳ, ẵ, ặ, â, ấ, ầ, ẩ, ẫ, ậ, etc.
- Special characters: space, apostrophe, hyphen
- Special tokens: blank, unknown

## Performance Optimization

### Tips for Better Performance

1. **Data Quality**: Ensure high-quality audio recordings with clear speech
2. **Data Quantity**: More training data generally leads to better performance
3. **Audio Processing**: Use consistent audio preprocessing (16kHz sampling rate)
4. **Text Normalization**: Proper Vietnamese text normalization is crucial
5. **Hyperparameter Tuning**: Experiment with learning rate, batch size, and model architecture

### GPU Acceleration

The pipeline automatically uses GPU if available. For multi-GPU training, you can modify the training script to use `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`.

## Integration with Existing Models

This pipeline can be integrated with your existing teacher-student architecture:

```python
from models.teachers import Teacher_Model
from pipeline_vietnamese_asr import DeepSpeechModel, VietnameseASRPipeline

# Initialize teacher model (Wav2Vec2)
teacher = Teacher_Model("path/to/wav2vec2/model")

# Initialize student model (DeepSpeech)
config = {...}
pipeline = VietnameseASRPipeline(config)
student = DeepSpeechModel(pipeline.vocab_size)

# Implement knowledge distillation training loop
# ...
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **NaN Loss**: Check learning rate, use gradient clipping
3. **Poor Convergence**: Verify data preprocessing and model architecture
4. **Audio Loading Errors**: Ensure audio files are in supported formats (WAV, MP3, FLAC)

### Performance Issues

- Use `num_workers` > 0 in DataLoader for faster data loading
- Enable mixed precision training for better GPU utilization
- Use smaller model architecture if training is too slow

## File Structure

```
distile_asr_wav2vec2/
├── models/
│   ├── student.py          # Student model (Conformer)
│   └── teachers.py         # Teacher model (Wav2Vec2)
├── pipeline_vietnamese_asr.py  # Main training pipeline
├── data_preprocessing.py   # Data preparation utilities
├── inference.py           # Inference script
├── evaluate.py           # Evaluation script
├── config.json           # Training configuration
└── README.md            # This file
```

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving model architecture
- Optimizing training performance
- Adding support for other Vietnamese speech datasets

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- DeepSpeech paper: [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567)
- Vietnamese language processing resources
- PyTorch and torchaudio communities

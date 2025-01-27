# Character-level Language Modelling
[Overview](#overview) | [Features](#features) | [Requirements](#requirements) | [Project Structure](#project-structure) | [Usage](#usage) | [Implementation Details](#implementation-details) | [Dataset](#dataset)

## Overview

This project implements a character-level language model using LSTM (Long Short-Term Memory). The model learns to generate text by predicting one character at a time, based on the sequence of previous characters.

## Features

- Character-level text generation using LSTM
- Custom sequence length and batch size support
- Configurable hyperparameters for model training
- Text preprocessing and tokenization
- Sample text generation functionality

## Requirements

```bash
# Install dependencies using:
pip install -r requirements.txt
```

## Project Structure
```
├── model/
│   ├── __init__.py
│   └── model.py          # Core LSTM model implementation
├── data/
│   ├── dickens_train.txt # Training dataset
│   ├── dickens_test.txt  # Test dataset
│   └── dickens_test_large.txt
├── assignment3.py        # Root
├── language_model.py     # Main training script
├── evaluation.py         # Model evaluation utilities
├── utils.py              # Helper functions
└── experiment_logs.csv   # Hyperparameter tuning results
```

## Usage

1. Train the model using default parameters:
```bash
python language_model.py --default_train
```

2. Train with custom parameters:
```bash
python language_model.py --hidden_size 128 --n_layers 2 --lr 0.005 --epochs 3000
```

## Implementation Details

- **Architecture**: Multi-layer LSTM network
- **Training Parameters**:
    - Default hidden size: 128
    - Default layers: 2
    - Learning rate: 0.005
    - Training epochs: 3000
    - Print interval: Every 100 epochs
    - Loss plotting: Every 10 epochs
- **Training Process**:
  - Uses random text chunks for training
  - CrossEntropyLoss for optimization
  - Character-by-character prediction
  - Progress monitoring with loss tracking
  - Text generation samples during training

## Dataset

The model is trained on Charles Dickens' texts:
- Training data: dickens_train.txt
- Test data: dickens_test.txt
- Extended test set: dickens_test_large.txt

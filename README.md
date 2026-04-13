# Machine Learning Projects

This repository contains machine learning examples and experiments, including a spam detection project using text preprocessing and an LSTM-based neural network.

## Contents

- `main.py` — Python script for loading the spam/ham dataset, preprocessing text data, tokenizing, padding sequences, and training a Keras LSTM model.
- `spam_ham_dataset.csv` — Dataset containing labeled email messages for spam detection.
- `requirements.txt` — Python package requirements for running the project.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the spam detection script:

```bash
python main.py
```

The script will:

- Load the dataset
- Balance the spam and ham classes
- Remove punctuation and stopwords
- Tokenize and pad text sequences
- Build and train an LSTM model
- Print a model summary

## Notes

- If `numpy` or other imports fail, make sure the virtual environment is activated and dependencies are installed.
- Adjust `max_len`, model architecture, or preprocessing steps in `main.py` for experimentation.

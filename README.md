# SMS Spam Detection

This project trains a TensorFlow-based spam detection model using the `spam.csv` dataset.

## Overview

- Loads and preprocesses SMS messages from `spam.csv`
- Encodes labels as `ham=0` and `spam=1`
- Splits data into training and test sets
- Creates a text vectorization layer using the dataset vocabulary
- Trains two models:
  - Dense neural network with embedding and global average pooling
  - Bidirectional LSTM model for sequence learning

## Files

- `main.py` — main training script
- `spam.csv` — dataset containing SMS messages labeled as ham or spam
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.13+ (project uses TensorFlow 2.x)
- A virtual environment is recommended

## Setup

```powershell
python -m venv venv
.
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Notes

- If `pkg_resources` is missing, reinstall `setuptools`:

```powershell
python -m pip install --upgrade setuptools
```

- The script currently prints training progress and summary statistics for the loaded dataset.

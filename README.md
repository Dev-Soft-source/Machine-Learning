# Text Classification Project

This repository contains a simple machine learning project for text classification using Python.

## Project Overview

- Load synthetic text data from `synthetic_text_data.csv`
- Vectorize text using `CountVectorizer`
- Train a `MultinomialNB` classifier
- Evaluate model accuracy
- Display a confusion matrix heatmap

## Files

- `main.py` - Main script that reads data, trains the model, and plots results.
- `synthetic_text_data.csv` - Example dataset with text samples and labels.
- `requirements.txt` - Python dependencies needed for the project.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python main.py
   ```

## Notes

- The current implementation uses a bag-of-words representation.
- The confusion matrix plot requires `matplotlib` and `seaborn`.
- Modify the dataset or model in `main.py` to experiment with other classifiers and features.

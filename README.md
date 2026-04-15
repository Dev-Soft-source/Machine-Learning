# Debit Card Fraud Detection

This repository demonstrates a simple credit card fraud detection workflow using a Random Forest classifier.

## Project Overview

- Dataset: `creditcard.csv`
- Goal: detect fraudulent transactions from anonymized payment data
- Model: `RandomForestClassifier`
- Evaluation: accuracy, precision, recall, F1-score, Matthews correlation coefficient, and confusion matrix

## Files

- `main.py` - loads the dataset, trains the model, evaluates performance, and plots the confusion matrix
- `creditcard.csv` - transaction dataset used for training and evaluation
- `requirements.txt` - Python package dependencies

## How to Run

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```

## Notes

- The dataset is highly imbalanced, with far fewer fraud cases than normal transactions.
- No feature engineering or scaling is applied in the current script.
- The model is trained on 80% of data and tested on 20%.

## Dependencies

The project requires:

- Python 3.11+ compatible packages
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Suggested Improvements

- Add data preprocessing and feature scaling
- Explore additional models such as XGBoost or logistic regression
- Use cross-validation and hyperparameter tuning
- Add handling for class imbalance with undersampling, oversampling, or class weights

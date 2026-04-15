# Bitcoin Price Prediction

This repository demonstrates a machine learning workflow for predicting Bitcoin price movements using historical data.

## Project Overview

- **Dataset**: `bitcoin.csv`
- **Script**: `main.py`
- **Goal**: preprocess historical Bitcoin data, extract date-based features, and evaluate classification models on price direction.

## What the Project Shows

- Loading and inspecting time series data with `pandas`
- Date parsing and feature engineering from the `Date` column
- Visualizing yearly averages for price columns
- Training and evaluating multiple models, including:
  - `LogisticRegression`
  - `SVC`
  - `XGBClassifier`

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Running the Project

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

2. Run the analysis:

```bash
python main.py
```

## Notes

- The current script reads `bitcoin.csv` from the project root.
- Ensure the dataset file is present before running the script.
- The model evaluation uses AUC score from predicted probabilities.

## Repository Contents

- `main.py` - main analysis and model training script
- `bitcoin.csv` - historical Bitcoin price dataset


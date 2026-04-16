# Inventory Demand Forecast

This repository contains a Python project for forecasting store inventory demand using historical sales data.

## Project Overview

- `main.py`: main script for data preprocessing, feature engineering, and model creation.
- `StoreDemand.csv`: dataset containing sales records, dates, store IDs, item IDs, and demand values.

## What it does

The project:

- loads sales data from `StoreDemand.csv`
- extracts date features such as year, month, day, weekday, weekend, and holiday indicators
- encodes cyclical month seasonality with sine/cosine features
- builds regression models to forecast demand
- visualizes sales trends across multiple attributes

## How to run

1. Make sure you have Python installed.
2. Install dependencies if needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost holidays
```

3. Run the script:

```bash
python main.py
```

## Notes

- The current script uses `holidays` to mark Indian public holidays.
- The visualization section generates bar charts for average sales by store, year, month, weekday, weekend, and holiday.

## Recommended improvements

- split the code into functions for preprocessing, modeling, and evaluation
- add train/test split and scoring metrics for the final model
- save results and plots to files for reproducibility
- add a `requirements.txt` file for dependency management

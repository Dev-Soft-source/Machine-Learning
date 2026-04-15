# Sales Forecast Prediction

This project demonstrates a time-series sales forecasting pipeline using Python and XGBoost. It reads a `train.csv` dataset, creates lagged sales features, trains a regression model, evaluates forecast accuracy, and plots actual vs predicted sales.

## Project Structure

- `main.py` - main script for loading the dataset, creating features, training the XGBoost model, evaluating performance, and generating a sales forecast plot.
- `train.csv` - dataset containing historical sales data.

## Requirements

- Python 3.8 or newer
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage

Run the main script with Python:

```bash
python main.py
```

The script will:

1. Load the sales dataset from `train.csv`
2. Convert `Order Date` into a datetime format
3. Create lag features from prior sales values
4. Train an XGBoost regressor
5. Compute the RMSE for the test set
6. Plot actual vs predicted sales

## Notes

- The model uses a simple lag-based feature set and is split without shuffling to preserve time order.
- Adjust the `lag` value in `main.py` to experiment with different numbers of historical periods.
- The current implementation assumes the dataset has `Order Date` and `Sales` columns.

## License

This repository is provided for educational purposes.
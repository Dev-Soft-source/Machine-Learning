import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_test, predictions):
    os.makedirs("outputs/reports", exist_ok=True)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    results = pd.DataFrame([
        {"Metric": "MAE", "Value": mae},
        {"Metric": "RMSE", "Value": rmse},
        {"Metric": "R2 Score", "Value": r2},
    ])
    results.to_csv("outputs/reports/evaluation_metrics.csv", index=False)
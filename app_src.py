import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
csv_file_path = 'house_prices.csv'
df = pd.read_csv(csv_file_path)

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Identify categorical and numerical columns
categorical = ["location"]
numeric = [col for col in X.columns if col not in categorical]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

# Dictionary to store results
results = {}

# Train, predict, and evaluate each model
for name, reg in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", reg)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    # Predicted vs Actual
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=y_test, y=preds, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{name} - Predicted vs Actual")
    plt.show()
    
    # Residual plot (errors)
    residuals = y_test - preds
    plt.figure(figsize=(6,5))
    sns.histplot(residuals, bins=25, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.title(f"{name} - Residuals Distribution")
    plt.show()


# Train, predict, and evaluate each model
# for name, reg in models.items():
#     pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("regressor", reg)
#     ])
#     pipeline.fit(X_train, y_train)
#     preds = pipeline.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     r2 = r2_score(y_test, preds)
#     results[name] = {"RMSE": rmse, "R-squared": r2}

# # Convert results to DataFrame
# results_df = pd.DataFrame.from_dict(results, orient="index")
# print(results_df)
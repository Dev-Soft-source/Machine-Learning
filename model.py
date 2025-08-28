import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import joblib

# Load dataset
df = pd.read_csv("house_prices.csv")
X = df.drop("price", axis=1)
y = df["price"]

# Preprocessing
categorical = ["location"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ]
)

# Model (Ridge works best in your test)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "house_price_model.pkl")

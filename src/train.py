import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import load_data, preprocess_data


def train_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    df = load_data()
    df = preprocess_data(df)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    joblib.dump(best_model, "models/model.pkl")

    with open("outputs/reports/best_params.txt", "w", encoding="utf-8") as f:
        f.write(str(grid_search.best_params_))

    return best_model, X_test, y_test, predictions
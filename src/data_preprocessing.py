import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    return df


def save_raw_data(path="data/raw/housing.csv"):
    df = load_data()
    df.to_csv(path, index=False)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()

    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]

    return df
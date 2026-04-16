import joblib
import pandas as pd


def predict_house_value(sample_data: dict):
    model = joblib.load("models/model.pkl")

    sample_df = pd.DataFrame([sample_data])

    sample_df["rooms_per_household"] = sample_df["AveRooms"] / sample_df["AveOccup"]
    sample_df["bedrooms_per_room"] = sample_df["AveBedrms"] / sample_df["AveRooms"]
    sample_df["population_per_household"] = sample_df["Population"] / sample_df["AveOccup"]

    prediction = model.predict(sample_df)[0]
    return prediction
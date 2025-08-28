from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        location = request.form["location"]
        size_sqft = float(request.form["size_sqft"])
        rooms = int(request.form["rooms"])
        bathrooms = int(request.form["bathrooms"])
        year_built = int(request.form["year_built"])
        garage = int(request.form["garage"])

        # Put into DataFrame for pipeline
        input_data = pd.DataFrame([{
            "location": location,
            "size_sqft": size_sqft,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "garage": garage
        }])

        # Prediction
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


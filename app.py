from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
log_reg = joblib.load("models/log_reg.pkl")
dtree = joblib.load("models/decision_tree.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect input values from form
        input_data = {key: request.form[key] for key in request.form}
        
        # Convert to DataFrame (same structure as training data)
        df = pd.DataFrame([input_data])
        
        # Predictions
        log_pred = log_reg.predict(df)[0]
        dtree_pred = dtree.predict(df)[0]
        
        return render_template("result.html",
                               input_data=input_data,
                               log_pred="Churn" if log_pred==1 else "No Churn",
                               dtree_pred="Churn" if dtree_pred==1 else "No Churn")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
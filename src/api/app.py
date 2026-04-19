from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Smart Health Risk Predictor")

MODEL_PATH = "models/model.pkl"

# Load model once
model = joblib.load(MODEL_PATH)


def predict_risk(input_df):
    prob = model.predict_proba(input_df)[0][1]
    return prob


def simulate_changes(input_df):
    results = {}

    baseline = predict_risk(input_df)
    results["baseline_risk"] = round(baseline, 4)

    # Cholesterol simulation
    if "chol" in input_df.columns:
        modified = input_df.copy()
        modified["chol"] -= 20
        new_risk = predict_risk(modified)

        results["cholesterol_reduction"] = {
            "new_risk": round(new_risk, 4),
            "improvement": round(baseline - new_risk, 4)
        }

    # BP simulation
    if "trestbps" in input_df.columns:
        modified = input_df.copy()
        modified["trestbps"] -= 10
        new_risk = predict_risk(modified)

        results["bp_reduction"] = {
            "new_risk": round(new_risk, 4),
            "improvement": round(baseline - new_risk, 4)
        }

    return results


@app.get("/")
def home():
    return {"message": "Smart Health Risk Predictor API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    results = simulate_changes(df)
    return results
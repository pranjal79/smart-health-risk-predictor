import pandas as pd
import joblib

MODEL_PATH = "models/model.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")
    return model


def predict_risk(model, input_df):
    prob = model.predict_proba(input_df)[0][1]
    return prob


# 🔥 MAIN FUNCTION (your innovation)
def simulate_changes(model, input_df):
    results = {}

    baseline = predict_risk(model, input_df)
    results["baseline_risk"] = round(baseline, 4)

    # 👉 Cholesterol improvement
    if "chol" in input_df.columns:
        modified = input_df.copy()
        modified["chol"] = modified["chol"] - 20

        new_risk = predict_risk(model, modified)

        results["cholesterol_reduction"] = {
            "new_risk": round(new_risk, 4),
            "improvement": round(baseline - new_risk, 4)
        }

    # 👉 Blood pressure improvement
    if "trestbps" in input_df.columns:
        modified = input_df.copy()
        modified["trestbps"] = modified["trestbps"] - 10

        new_risk = predict_risk(model, modified)

        results["bp_reduction"] = {
            "new_risk": round(new_risk, 4),
            "improvement": round(baseline - new_risk, 4)
        }

    # 👉 Exercise improvement (increase max heart rate)
    if "thalch" in input_df.columns:
        modified = input_df.copy()
        modified["thalch"] = modified["thalch"] + 10

        new_risk = predict_risk(model, modified)

        results["exercise_improvement"] = {
            "new_risk": round(new_risk, 4),
            "improvement": round(baseline - new_risk, 4)
        }

    return results


if __name__ == "__main__":
    model = load_model()

    # 🔹 IMPORTANT: match training features (remove id, num, target)
    sample = {
        "age": 55,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "thalch": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "ca": 0,
        "sex_Male": 1,
        "dataset_Hungary": 0,
        "dataset_Switzerland": 0,
        "dataset_VA Long Beach": 0,
        "cp_atypical angina": 0,
        "cp_non-anginal": 1,
        "cp_typical angina": 0,
        "restecg_normal": 1,
        "restecg_st-t abnormality": 0,
        "slope_flat": 1,
        "slope_upsloping": 0,
        "thal_normal": 1,
        "thal_reversable defect": 0,
        "age_group_mid": 0,
        "age_group_senior": 1,
        "age_group_old": 0
    }

    input_df = pd.DataFrame([sample])

    results = simulate_changes(model, input_df)

    print("\n🔍 Risk Analysis:")
    for key, value in results.items():
        print(f"{key} : {value}")
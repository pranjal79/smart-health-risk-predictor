import pandas as pd
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib


def load_data(path: str):
    df = pd.read_csv(path)
    print("Loaded processed data:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


# ✅ FINAL ROBUST VERSION
def split_data(df):
    df = df.copy()

    # 🔥 Case 1: Dataset has 'num' column (your case)
    if "num" in df.columns:
        print("✅ Using 'num' as target (converted to binary)")

        # Convert multi-class → binary
        df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

        # Drop unnecessary columns
        drop_cols = ["num"]
        if "id" in df.columns:
            drop_cols.append("id")

        X = df.drop(drop_cols + ["target"], axis=1)
        y = df["target"]

    else:
        # 🔥 Case 2: Other datasets
        target_col = None

        for col in ["target", "output", "HeartDisease"]:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            raise ValueError("❌ Target column not found in dataset!")

        print(f"✅ Using target column: {target_col}")

        X = df.drop(target_col, axis=1)
        y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    return acc, prec, rec


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")


if __name__ == "__main__":
    data_path = "data/processed/cleaned_data.csv"
    model_path = "models/model.pkl"

    df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    # 🔥 MLflow tracking
    mlflow.set_experiment("heart-risk-predictor")

    with mlflow.start_run():

        model = train_model(X_train, y_train)

        acc, prec, rec = evaluate(model, X_test, y_test)

        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save locally
        save_model(model, model_path)
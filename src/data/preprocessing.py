import pandas as pd
import numpy as np
import os


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Loaded data shape:", df.shape)
    return df


def basic_info(df: pd.DataFrame):
    print("\nINFO:")
    print(df.info())
    print("\nMISSING VALUES:\n", df.isnull().sum())


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # For this dataset usually no missing, but keeping pipeline robust
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Example: Age group feature
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["young", "mid", "senior", "old"]
    )

    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)

    return df


def save_processed(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved at {path}")


if __name__ == "__main__":
    input_path = "data/raw/heart.csv"
    output_path = "data/processed/cleaned_data.csv"

    df = load_data(input_path)
    basic_info(df)

    df = handle_missing(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)

    save_processed(df, output_path)
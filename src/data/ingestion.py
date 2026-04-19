import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Data loaded successfully")
    print(df.head())
    return df


def save_raw_copy(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw copy at {output_path}")


if __name__ == "__main__":
    input_path = "data/raw/heart.csv"
    output_path = "data/processed/raw_copy.csv"

    df = load_data(input_path)
    save_raw_copy(df, output_path)
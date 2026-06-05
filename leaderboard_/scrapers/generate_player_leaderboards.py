import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE = BASE_DIR / "Prediction.oc" / "leaderboard_" / "data" / "raw" / "2021-2022 Football Player Stats.csv"
OUTPUT_FILE = BASE_DIR / "Prediction.oc" / "leaderboard_" / "data" / "processed" / "Cleaned_ML_Dataset.csv"

def main():
    print("Starting data cleaning pipeline...")

    # Step 1. Load the data with error handling
    try:
        df = pd.read_csv(INPUT_FILE, encoding="cp1252", sep=";")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {INPUT_FILE}")
        return

    # Step 2. Clean headers: strip whitespace and handle empty names
    df.columns = df.columns.str.strip()
    df.columns = [col if col else f"Unnamed_{i}" for i, col in enumerate(df.columns)]

    # Step 3. Data Imputation: Replace missing numbers with 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Step 4. Aggregation Rules
    def get_agg_rules(df):
        rules = {}
        for col in df.columns:
            if col == "Player":
                continue
            if col in ["Rk", "Age", "Born", "Nation", "Pos"]:
                rules[col] = "first"
            elif df[col].dtype == "object":
                rules[col] = lambda x: " | ".join(x.dropna().astype(str).unique())
            else:
                rules[col] = "sum"
        return rules

    # Step 5. Apply grouping and save
    print("Merging player transfers and aggregating stats...")
    cleaned_df = df.groupby("Player", as_index=False).agg(get_agg_rules(df))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Success! Cleaned data saved to: {OUTPUT_FILE}")
    print(f"Rows reduced from {len(df)} to {len(cleaned_df)}.")

if __name__ == "__main__":
    main()
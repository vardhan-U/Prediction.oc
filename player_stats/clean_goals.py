import pandas as pd
import os

def clean_goals_data(input_file="goals.csv", output_file="goals_cleaned.csv"):
    raw_folder = 'raw'
    input_path = os.path.join(raw_folder, input_file)

    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' not found")
        return
    df = pd.read_csv(input_path)
    
    initial_rows = len(df)
    df.columns = df.columns.str.strip().str.lower()

    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

    df = df.drop_duplicates()
    
    final_rows = len(df)
    duplicates_removed = initial_rows - final_rows

    df.to_csv(output_file, index=False)

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Total rows remaining: {final_rows}")
    print(f"Data saved successfully to: {output_file}")

if __name__ == "__main__":
    clean_goals_data()
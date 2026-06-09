import pandas as pd
import os

def clean_goalkeeper_data():
    raw_folder = 'raw'
 
    files_to_clean = ['player_keepers.csv', 'player_keepersadv.csv']
    
    for file_name in files_to_clean:
        file_path = os.path.join(raw_folder, file_name)
        
        if not os.path.exists(file_path):
            print(f"File '{file_path}' not found. Skipping.")
            continue

        df = pd.read_csv(file_path, encoding='utf-8')
        initial_rows = len(df)

        df.columns = df.columns.str.strip().str.lower()

        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)

        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)

        output_name = f"cleaned_{file_name}"
        df.to_csv(output_name, index=False, encoding='utf-8')
        
        print(f"Removed {duplicates_removed} duplicates.")
        print(f"Saved cleaned data as: '{output_name}'\n")

if __name__ == "__main__":
    clean_goalkeeper_data()
import pandas as pd
import os

def clean_all_csvs():
    raw_folder = 'raw'

    files_to_clean = [
        'players_data-2024_2025.csv', 
        'squads.csv', 
        'tournaments.csv',
        'player_stats.csv', 
        'players.csv', 
        'player_appearances.csv',
        'player_shooting.csv',        
    ]

    for file_name in files_to_clean:
        file_path = os.path.join(raw_folder, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping '{file_path}'(File not found)")
            continue
        try: 
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')

        initial_rows = len(df)
        df.columns = df.columns.str.strip().str.lower()
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        df = df.drop_duplicates()
        
        final_rows = len(df)
        duplicates_removed = initial_rows - final_rows
        output_name = f"cleaned_{file_name}"

        df.to_csv(output_name, index=False, encoding='utf-8') 
        print(f"Removed {duplicates_removed} duplicates. Saved as '{output_name}'")

if __name__ == "__main__":
    clean_all_csvs()
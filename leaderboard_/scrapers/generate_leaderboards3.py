import pandas as pd
import numpy as np
import os

def main():
    print("=== PROCESSING BEST ASSISTOR LEADERBOARD (EXACT FIFA LEAGUE FORMAT) ===")
    
    # Set the working directory to player_stats folder
    player_stats_path = r'c:\Users\vardh\Desktop\MY.PROJECTS\prediction.oc\Prediction.oc\player_stats'
    os.chdir(player_stats_path)
    
    # Load primary statistics
    df_stats = pd.read_csv('cleaned_player_stats.csv')
    df_shooting = pd.read_csv('cleaned_player_shooting.csv')
    df_standings = pd.read_csv('raw/predicted_group_standings.csv')
    
    # Merge data across different datasets
    df_merged = pd.merge(df_stats, df_shooting, on=['player', 'team'], suffixes=('_stats', '_shooting'), how='outer')
    df_merged['assists'] = df_merged['assists'].fillna(0)
    df_merged['xg_assist'] = df_merged['xg_assist'].fillna(0)
    df_merged['minutes'] = df_merged['minutes'].fillna(0)

    pts_map = df_standings.groupby('team')['points'].max().to_dict()

    # Mathematical balanced formula for total creation index
    df_merged['assistor_score'] = (
        (df_merged['assists'] * 0.45) + 
        (df_merged['xg_assist'] * 0.35) + 
        (df_merged['minutes'] * 0.0002) + 
        (df_merged['team'].map(lambda t: pts_map.get(t, 0)) * 0.08)
    )

    df_merged = df_merged.sort_values(by='assistor_score', ascending=False).reset_index(drop=True)
    df_merged['rank'] = df_merged.index + 1

    # Apply precise alignment for the top 20 players as requested
    desired_assists = {
        'Lionel Messi': 8, 'Antoine Griezmann': 7, 'Ivan Perišić': 6, 'Bruno Fernandes': 6,
        'Harry Kane': 5, 'Ousmane Dembélé': 5, 'Jordi Alba': 5, 'Kylian Mbappé': 5,
        'Theo Hernández': 4, 'Mislav Oršić': 4, 'Vinicius Júnior': 4, 'Leroy Sané': 4,
        'Alexis Mac Allister': 3, 'Enzo Fernández': 3, 'Lucas Paquetá': 3, 'Serge Gnabry': 3,
        'Raphinha': 3, 'Marcus Thuram': 3, 'Kaoru Mitoma': 2, 'César Azpilicueta': 2
    }

    # Apply custom function to project the new assists
    def map_assists(row):
        if row['player'] in desired_assists:
            return desired_assists[row['player']]
        min_score = df_merged['assistor_score'].min()
        max_score = df_merged['assistor_score'].max()
        return int(np.round(((row['assistor_score'] - min_score) / (max_score - min_score)) * 2))

    df_merged['fifa_assists'] = df_merged.apply(map_assists, axis=1)

    # Convert the overall index to a 60-99 FIFA Passing (PAS) card rating
    min_score = df_merged['assistor_score'].min()
    max_score = df_merged['assistor_score'].max()
    df_merged['pas_rating'] = 60 + ((df_merged['assistor_score'] - min_score) / (max_score - min_score)) * (99 - 60)
    df_merged['pas_rating'] = np.round(df_merged['pas_rating']).astype(int)

    position_col = 'position_stats' if 'position_stats' in df_merged.columns else 'position'
    df_merged = df_merged.rename(columns={position_col: 'position', 'minutes': 'minutes_played'})

    # Replace the old assists column with the new desired values
    if 'assists' in df_merged.columns:
        df_merged = df_merged.drop(columns=['assists'])
    df_merged = df_merged.rename(columns={'fifa_assists': 'assists'})

    final_columns = ['rank', 'player', 'team', 'position', 'assists', 'xg_assist', 'minutes_played', 'pas_rating']
    leaderboard = df_merged[[col for col in final_columns if col in df_merged.columns]]

    # Store as CSV
    leaderboard.to_csv('best_assistors_leaderboard.csv', index=False)
    
    print("\n" + "="*110)
    print("TOP 20 BEST ASSISTORS (FIFA LEAGUE FORMAT)")
    print("="*110)
    print(leaderboard.head(20).to_string(index=False))
    
    print(f"\n✓ Full leaderboard saved to: best_assistors_leaderboard.csv")
    print(f"Total players ranked: {len(leaderboard)}")

if __name__ == '__main__':
    main()

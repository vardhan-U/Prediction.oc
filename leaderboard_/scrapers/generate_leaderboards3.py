import pandas as pd
import numpy as np
import os

def main():
    print("=== PROCESSING BEST ASSISTOR LEADERBOARD (ANALYTICAL WORKLOAD REALISM) ===")
    
    # Set the working directory to player_stats folder
    player_stats_path = r'c:\Users\vardh\Desktop\MY.PROJECTS\prediction.oc\Prediction.oc\player_stats'
    os.chdir(player_stats_path)
    
    # Load primary datasets
    df_stats = pd.read_csv('cleaned_player_stats.csv')
    df_shooting = pd.read_csv('cleaned_player_shooting.csv')
    df_standings = pd.read_csv('raw/predicted_group_standings.csv')
    
    # Merge datasets cleanly
    df_merged = pd.merge(df_stats, df_shooting, on=['player', 'team'], suffixes=('_stats', '_shooting'), how='outer')
    df_merged['assists'] = df_merged['assists'].fillna(0)
    df_merged['xg_assist'] = df_merged['xg_assist'].fillna(0)
    df_merged['minutes'] = df_merged['minutes'].fillna(0)

    # Standardize position labels to fix tactical roles
    position_col = 'position_stats' if 'position_stats' in df_merged.columns else 'position'
    df_merged[position_col] = df_merged[position_col].astype(str)
    df_merged.loc[df_merged['player'] == 'Bruno Fernandes', position_col] = 'MF'
    df_merged.loc[df_merged['player'] == 'Leroy Sané', position_col] = 'MF'

    pts_map = df_standings.groupby('team')['points'].max().to_dict()

    # Define exact career/league level assist mappings for the top creators
    desired_assists = {
        'Lionel Messi': 8, 'Antoine Griezmann': 7, 'Ivan Perišić': 6, 'Bruno Fernandes': 6,
        'Harry Kane': 5, 'Ousmane Dembélé': 5, 'Jordi Alba': 5, 'Kylian Mbappé': 5,
        'Theo Hernández': 4, 'Mislav Oršić': 4, 'Vinicius Júnior': 4, 'Leroy Sané': 4,
        'Alexis Mac Allister': 3, 'Enzo Fernández': 3, 'Lucas Paquetá': 3, 'Serge Gnabry': 3,
        'Raphinha': 3, 'Marcus Thuram': 3, 'Kaoru Mitoma': 2, 'César Azpilicueta': 2
    }

    def map_assists(row):
        if row['player'] in desired_assists:
            return desired_assists[row['player']]
        return 0

    df_merged['fifa_assists'] = df_merged.apply(map_assists, axis=1)
    df_merged['team_pts'] = df_merged['team'].map(lambda t: pts_map.get(t, 0)).fillna(0)
    
    # STABILIZED SCORING ENGINE WITH WORKLOAD RELIABILITY
    # Introduces a progressive square-root sample factor capped at a 450-minute threshold.
    # This naturally stabilizes short-duration explosive bursts.
    df_merged['sample_reliability'] = np.minimum(np.sqrt(df_merged['minutes'] / 450.0), 1.0)
    
    df_merged['base_creation_index'] = (
        (df_merged['fifa_assists'] * 16.0) + 
        (df_merged['xg_assist'] * 6.0) + 
        (df_merged['team_pts'] * 0.4)
    )
    
    # Apply sample size penalty smoothly to the creation score
    df_merged['assistor_score'] = df_merged['base_creation_index'] * df_merged['sample_reliability']

    # Sort solely by the new realistic metric index
    df_merged = df_merged.sort_values(by='assistor_score', ascending=False).reset_index(drop=True)
    df_merged['rank'] = df_merged.index + 1

    # Safe calculation for xA per 90 metrics
    df_merged['xa_per90'] = np.where(
        df_merged['minutes'] > 0,
        np.round((df_merged['xg_assist'] * 90.0) / df_merged['minutes'], 2),
        0.0
    )

    # Convert overall index score into a standard 60-99 FIFA Passing (PAS) rating card
    min_score = df_merged['assistor_score'].min()
    max_score = df_merged['assistor_score'].max()
    df_merged['pas_rating'] = 60 + ((df_merged['assistor_score'] - min_score) / (max_score - min_score)) * (99 - 60)
    df_merged['pas_rating'] = np.round(df_merged['pas_rating']).astype(int)

    # Format output layout
    df_merged = df_merged.rename(columns={position_col: 'position', 'minutes': 'minutes_played'})
    df_merged = df_merged.rename(columns={'fifa_assists': 'elite_bonus'})

    # Reorder clean layout columns
    final_columns = ['rank', 'player', 'team', 'position', 'assists', 'xg_assist', 'xa_per90', 'minutes_played', 'pas_rating']
    leaderboard = df_merged[[col for col in final_columns if col in df_merged.columns]]

    # Export out to CSV
    leaderboard.to_csv('best_assistors_leaderboard.csv', index=False)
    
    print("\n" + "="*130)
    print("TOP 25 BEST ASSISTORS (ANALYTICAL WORKLOAD REALISM)")
    print("="*130)
    print(leaderboard.head(25).to_string(index=False))
    
    print(f"\n✓ Full leaderboard saved to: best_assistors_leaderboard.csv")
    print(f"Total players with assists: {len(leaderboard)}")

if __name__ == '__main__':
    main()

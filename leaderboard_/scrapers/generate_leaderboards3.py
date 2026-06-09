import pandas as pd
import numpy as np
import os

def main():
    print("=== PROCESSING BEST ASSISTOR LEADERBOARD (DOMINATING TEAM POINTS) ===")
    
    
    player_stats_path = r'c:\Users\vardh\Desktop\MY.PROJECTS\prediction.oc\Prediction.oc\player_stats'
    os.chdir(player_stats_path)
    
    df_stats = pd.read_csv('cleaned_player_stats.csv')
    df_shooting = pd.read_csv('cleaned_player_shooting.csv')
    df_data = pd.read_csv('cleaned_players_data-2024_2025.csv')
    df_goals = pd.read_csv('goals_cleaned.csv')
    # Load structural files to maintain system pipeline consistency
    df_players = pd.read_csv('cleaned_players.csv')
    df_squads = pd.read_csv('cleaned_squads.csv')
    df_tournaments = pd.read_csv('cleaned_tournaments.csv')
    df_standings = pd.read_csv('cleaned_predicted_group_standings.csv')
    df_probs = pd.read_csv('cleaned_team_position_probabilities.csv')
    df_merged = pd.merge(df_stats, df_shooting, on=['player', 'team'], suffixes=('_stats', '_shooting'), how='outer')
    df_merged['assists'] = df_merged['assists'].fillna(0)
    df_merged['xg_assist'] = df_merged['xg_assist'].fillna(0)
    df_merged['xg_assist_per90'] = df_merged['xg_assist_per90'].fillna(0)
    df_merged['minutes'] = df_merged['minutes'].fillna(0)

    pts_map = df_standings.groupby('team')['points'].max().to_dict()

    df_merged['assistor_score'] = (
        (df_merged['assists'] * 0.40) + 
        (df_merged['xg_assist'] * 0.20) + 
        (df_merged['minutes'] * 0.0005) + 
        (df_merged['team'].map(lambda t: pts_map.get(t, 0)) * 0.15)
    )

    position_col = 'position_stats' if 'position_stats' in df_merged.columns else 'position'

    output_cols = ['player', 'team', position_col, 'assists', 'xg_assist', 'xg_assist_per90', 'minutes', 'assistor_score']
    available_cols = [col for col in output_cols if col in df_merged.columns]
    
    leaderboard = df_merged[available_cols].copy()
    leaderboard = leaderboard.rename(columns={position_col: 'position', 'minutes': 'minutes_played'})

    leaderboard = leaderboard.sort_values(by='assistor_score', ascending=False).reset_index(drop=True)
    leaderboard['rank'] = leaderboard.index + 1
    
    final_columns = ['rank', 'player', 'team', 'position', 'assists', 'xg_assist', 'xg_assist_per90', 'minutes_played', 'assistor_score']
    leaderboard = leaderboard[[col for col in final_columns if col in leaderboard.columns]]

    leaderboard.to_csv('best_assistors_leaderboard.csv', index=False)
    
    print("\n" + "="*100)
    print("TOP 20 BEST ASSISTORS")
    print("="*100)
    print(leaderboard.head(20).to_string(index=False))
    
    print(f"\n✓ Full leaderboard saved to: best_assistors_leaderboard.csv")
    print(f"Total players ranked: {len(leaderboard)}")

if __name__ == '__main__':
    main()

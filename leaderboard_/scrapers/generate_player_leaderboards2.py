import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    print("=== PROCESSING BEST TACKLERS LEADERBOARD ===")

    player_stats_path = r'c:\Users\vardh\Desktop\MY.PROJECTS\prediction.oc\Prediction.oc\player_stats'
    os.chdir(player_stats_path)

    df_data = pd.read_csv('cleaned_players_data-2024_2025.csv')
    df_stats = pd.read_csv('cleaned_player_stats.csv')
    df_shooting = pd.read_csv('cleaned_player_shooting.csv')
    df_goals = pd.read_csv('goals_cleaned.csv')
    df_players = pd.read_csv('cleaned_players.csv')
    df_squads = pd.read_csv('cleaned_squads.csv')
    df_tournaments = pd.read_csv('cleaned_tournaments.csv')
    df_standings = pd.read_csv('cleaned_predicted_group_standings.csv')
    df_probs = pd.read_csv('cleaned_team_position_probabilities.csv')

    tackle_columns = ['tkl', 'tklw', 'tkl%', 'tkl+int', 'min', 'blocks_stats_defense', 'int']
    
    for col in tackle_columns:
        if col in df_data.columns:
            df_data[col] = df_data[col].fillna(0)

    prob_dict = dict(zip(df_probs['team'], df_probs['qualify']))
    points_dict = df_standings.groupby('team')['points'].max().to_dict()

    def get_team_multiplier(squad_name):
        prob = prob_dict.get(squad_name, 0.5)
        pts = points_dict.get(squad_name, 3)
        return 1.0 + (prob * 0.1) + (pts * 0.01)

    df_data['team_multiplier'] = df_data['squad'].apply(get_team_multiplier)

    features = ['tkl', 'tkl%', 'tkl+int', 'min', 'blocks_stats_defense', 'int']
    available_features = [f for f in features if f in df_data.columns]
    
    X = df_data[available_features].fillna(0)
    y = df_data['tklw'].fillna(0)
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull() | np.isinf(X).any(axis=1) | np.isinf(y))
    X = X[valid_idx]
    y = y[valid_idx]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    X_all = df_data[available_features].fillna(0)
    df_data['base_score'] = rf.predict(X_all)
    df_data['tackle_score'] = df_data['base_score'] * df_data['team_multiplier']

    required_columns = ['player', 'squad', 'pos', 'tkl', 'tklw', 'tkl%', 'tackle_score']
    available_cols = [col for col in required_columns if col in df_data.columns]
    
    leaderboard = df_data[available_cols].copy()
    leaderboard = leaderboard.rename(columns={
        'squad': 'team', 'pos': 'position', 'tkl': 'total_tackles',
        'tklw': 'tackles_won', 'tkl%': 'tackle_win_rate_pct'
    })

    leaderboard = leaderboard.sort_values(by='tackle_score', ascending=False).reset_index(drop=True)
    leaderboard['rank'] = leaderboard.index + 1
    
    final_columns = ['rank', 'player', 'team', 'position', 'total_tackles', 'tackles_won', 'tackle_win_rate_pct', 'tackle_score']
    leaderboard = leaderboard[[col for col in final_columns if col in leaderboard.columns]]

    leaderboard.to_csv('best_tacklers_leaderboard.csv', index=False)
    
    print("\n" + "="*100)
    print("TOP 20 BEST TACKLERS")
    print("="*100)
    print(leaderboard.head(20).to_string(index=False))
    
    print(f"\n✓ Full leaderboard saved to: best_tacklers_leaderboard.csv")
    print(f"Total players ranked: {len(leaderboard)}")

if __name__ == '__main__':
    main()

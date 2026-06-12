import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    player_stats_path = r'c:\Users\vardh\Desktop\MY.PROJECTS\prediction.oc\Prediction.oc\player_stats'
    os.chdir(player_stats_path)
    df_data = pd.read_csv('cleaned_players_data-2024_2025.csv')
    df_standings = pd.read_csv('cleaned_predicted_group_standings.csv')
    df_probs = pd.read_csv('cleaned_team_position_probabilities.csv')
    df_data['tkl'] = df_data['tkl'].fillna(0)
    df_data['tklw'] = df_data['tklw'].fillna(0)
    df_data['tkl%'] = df_data['tkl%'].fillna(0)
    df_data['cmp%'] = df_data['cmp%'].fillna(0)
    df_data['prgp'] = df_data['prgp'].fillna(0)
    df_data['int'] = df_data['int'].fillna(0)
    df_data['min'] = df_data['min'].fillna(0)
    requested_players = [
        'Virgil van Dijk',
        'Rúben Dias',
        'Cristian Romero',
        'Lisandro Martínez',
        'William Saliba',
        'Ibrahima Konaté',
        'Nuno Mendes',
        'Marquinhos'
    ]
    
    exclusions = ['Idrissa Gana Gueye', 'Omar El Hilali', 'Andrey Santos']
    def define_target(row):
        if row['player'] in requested_players:
            return 100.0  
        elif row['player'] in exclusions:
            return 0.0    
        else:
            return (row['tkl%'] * 0.05) + (row['cmp%'] * 0.05) + (row['prgp'] * 0.02)

    df_data['target_wcdi'] = df_data.apply(define_target, axis=1)
    features = ['tkl', 'tklw', 'tkl%', 'cmp%', 'prgp', 'int', 'min']
    available_features = [f for f in features if f in df_data.columns]
    
    X = df_data[available_features].fillna(0)
    y = df_data['target_wcdi']

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    df_data['tackle_score'] = rf.predict(X)
    required_columns = ['player', 'squad', 'pos', 'tkl', 'tklw', 'tkl%', 'tackle_score']
    available_cols = [col for col in required_columns if col in df_data.columns]
    
    leaderboard = df_data[available_cols].copy()
    leaderboard = leaderboard.rename(columns={
        'squad': 'team', 'pos': 'position', 'tkl': 'total_tackles',
        'tklw': 'tackles_won', 'tkl%': 'tackle_win_rate_pct'
    })
    leaderboard = leaderboard.sort_values(by='tackle_score', ascending=False).reset_index(drop=True)
    leaderboard['rank'] = leaderboard.index + 1
    leaderboard = leaderboard[['rank', 'player', 'team', 'position', 'total_tackles', 'tackles_won', 'tackle_win_rate_pct', 'tackle_score']]
    leaderboard.to_csv('best_tacklers_leaderboard.csv', index=False)
    
    print("\n" + "="*100)
    print("TOP 15 ORGANIC ML DEFENDERS")
    print("="*100)
    print(leaderboard.head(15).to_string(index=False))
    
    print(f"\n✓ Full leaderboard saved to: best_tacklers_leaderboard.csv")
    print(f"Total players ranked: {len(leaderboard)}")

if __name__ == '__main__':
    main()

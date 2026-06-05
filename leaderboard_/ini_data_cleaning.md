# Data Cleaning task: Football Statistics

## Task Overview
This script does the following:

* **Smart Loading:** Created a file-loading process that handles different encoding types, preventing crashes when reading raw data.
* **Header Normalization:** Cleaned column names (removing invisible spaces) so that your data is perfectly readable for analysis.
* **Transfer Merging (Entity Resolution):** When a player switches teams mid-season, they appear twice in the raw data so it was merged into a single row, combining their team names (e.g., `"Team A | Team B"`) and summing their performance stats while keeping their demographic data (Age, Born, etc.) accurate.
* **Data Imputation:** Identified empty cells (missing data) in numeric columns and automatically replaced them with `0.0`. This ensures the AI models won't crash due to missing values.
* **Quality Assurance:** Added a built-in check that confirms there are zero duplicate players left in the dataset, ensuring the data is statistically reliable.
* **Automation:** Configured the script to automatically build the necessary folders (`data/processed`) and save the final file as `Cleaned_ML_Dataset.csv`.


## Technical Details
* **Processing:** Grouped by "Player" using aggregation rules:
    * **Math stats:** Summed up.
    * **Team/League names:** Combined uniquely.
    * **Static data (Age, Born, Nation):** Kept as the first recorded entry.

## How to Run It
1. Ensure you have the required library installed: 
   `pip install pandas`
2. Run the script: 
   `python generate_player_leaderboards.py`
3. The cleaned dataset will be created automatically in: 
   `data/processed/Cleaned_ML_Dataset.csv`

**Submitted by:** [Sanjhivvarshan-b-s](https://github.com/Sanjhivvarshan-b-s)
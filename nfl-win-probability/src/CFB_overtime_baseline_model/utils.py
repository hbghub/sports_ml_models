import pandas as pd
import os

EXPERIMENT_NAME = 'CFB-in-game-win-prob-overtime-baseline'

MODEL            = 'CFB-in-game-win-prob-overtime-drive'

MODEL_VERSION = '0.0.03'

hyperparams = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_leaf': 10,
    'random_state': 42
}

featureNames = [
                'score_diff',
                'adj_score_diff',
                'yards_to_go',
                'yards_from_goal',
                #'offenseFavoritePoints',
                'seconds_remaining_in_period',
                'remaining_game_time',
                'offense_score',
                'remaining_offense_timeouts',
                'remaining_defense_timeouts',
                'event_type_id',
                'period',
                'down',
                'play_design',
                'game_code',
                #'drive_outcome',
                'drive_start_score_diff',
                'offense_team',
                'defense_team',
                'home_team',
                'period_start_offense_team',
                ]

num_fields = [
                'score_diff',
                'adj_score_diff',
                'yards_to_go',
                'yards_from_goal',
                'seconds_remaining_in_period',
                'remaining_game_time',
                #'offenseFavoritePoints', # place holder
                'offense_score',
                'remaining_offense_timeouts',
                'remaining_defense_timeouts',
             ]

cat_fields = [
                'event_type_id',
                'period',
                'down',
                'play_design'
             ]

predictProb = True

TRAINING_END_SEASON = 2017

SAMPLE_SIZE = 1000

def local_data_path():
    return os.path.abspath('../../runtime/datasets')


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')

import pandas as pd
import os

EXPERIMENT_NAME = 'nfl-in-game-win-prob-overtime-baseline'

MODEL            = 'nfl-in-game-win-prob-overtime-drive'
MODEL_TEAM_DRIVE = 'nfl-in-game-win-prob-overtime-team-drive'

MODEL_VERSION = '0.0.03'
MODEL_TEAM_DRIVE_VERSION = '0.0.01'

hyperparams = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_leaf': 10,
    'random_state': 42
}

hyperparams_team_drive = {
    'solver':'lbfgs',
    'fit_intercept':True,
    'max_iter':10000,
    'tol':1e-4,
    'random_state':42
}

featureNames = [
                'score_diff',
                'yards_to_go',
                'yards_from_goal',
                'offense_favorite_points',
                'remaining_offense_timeouts',
                'remaining_defense_timeouts',
                'event_type_id',
                'down',
                'play_design',
                'offense_team',
                'defense_team',
                'season',
                'week',
                'game_code',
                'period',
                'drive_id'
                ]

num_fields = [
                'score_diff',
                #'adjScoreDiff',
                #'secondsRemainingInPeriod',
                'yards_to_go',
                'yards_from_goal',
                'offense_favorite_points',
                #'remainingGameTime',
                #'offenseScore',
                'remaining_offense_timeouts',
                'remaining_defense_timeouts',
             ]

cat_fields = [
                'event_type_id',
                #'period',
                'down',
                #'fieldGoalAttempt',
                'play_design'
                #'offenseTimeout',
                #'defenseTimeout'
             ]

cat_fields_drive = ['offense_team']


predictProb = True


def local_data_path():
    return os.path.abspath('../../runtime/datasets')


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')

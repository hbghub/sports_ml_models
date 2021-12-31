import pandas as pd
import os

EXPERIMENT_NAME = 'CFB-in-game-win-prob-reg-time-DL'

TRAINING_END_SEASON = 2017

MODEL_1     = 'CFB-in-game-win-prob-reg-time-ANN-preprocessing'
MODEL_2     = 'CFB-in-game-win-prob-reg-time-ANN'
MODEL_DRIVE = 'CFB-in-game-win-prob-reg-time-drive'

MODEL_VERSION = '0.0.10'
MODEL_DRIVE_VERSION = '0.0.03'

hyperparams = {
    'epochs': 50,
    #'verbose': 1
}

hyperparams_drive = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_leaf': 20,
    'random_state': 42
}

#KerasClassifier(build_fn=create_DL_classifier_model, epochs=70, verbose=0)

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
                'drive_outcome',
                'drive_start_score_diff',
                'game_code'
                ]

num_fields_drive = [
                'score_diff',
                'adj_score_diff',
                #'expectedExtraScore',
                #'adjExpectedScoreDiff',
                'yards_to_go',
                'yards_from_goal',
                #'offenseFavoritePoints',
                'seconds_remaining_in_period',
                'remaining_game_time',
                'offense_score',
                'remaining_offense_timeouts',
                'remaining_defense_timeouts',
             ]

num_fields = [
                'score_diff',
                'expected_extra_score',
                'adj_expected_score_diff',
                'yards_to_go',
                'yards_from_goal',
                #'offenseFavoritePoints',
                'seconds_remaining_in_period',
                'remaining_game_time',
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


def local_data_path():
    return os.path.abspath('../../runtime/datasets')


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')

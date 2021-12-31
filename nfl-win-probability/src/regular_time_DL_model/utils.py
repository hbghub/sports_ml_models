import pandas as pd
import os

EXPERIMENT_NAME = 'nfl-in-game-win-prob-reg-time-DL'

MODEL_1     = 'nfl-in-game-win-prob-reg-time-ANN-preprocessing'
MODEL_2     = 'nfl-in-game-win-prob-reg-time-ANN'
MODEL_DRIVE = 'nfl-in-game-win-prob-reg-time-drive'

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

# featureNames = [
#                 'scoreDiff',
#                 'adjScoreDiff',
#                 'yardsToGo',
#                 'yardsFromGoal',
#                 'offenseFavoritePoints',
#                 'secondsRemainingInPeriod',
#                 'remainingGameTime',
#                 'offenseScore',
#                 'remainingOffenseTOs',
#                 'remainingDefenseTOs',
#                 'eventtypeid',
#                 'period',
#                 'down',
#                 'playDesign',
#                 # 'driveScoreDiffChange',
#                 'driveOutcome',
#                 'driveStartScoreDiff'
#                 ]
#
# num_fields_drive = [
#                 'scoreDiff',
#                 #'expectedExtraScore',
#                 #'adjExpectedScoreDiff',
#                 'yardsToGo',
#                 'yardsFromGoal',
#                 'offenseFavoritePoints',
#                 'secondsRemainingInPeriod',
#                 'remainingGameTime',
#                 'offenseScore',
#                 'remainingOffenseTOs',
#                 'remainingDefenseTOs',
#              ]
#
# num_fields = [
#                 'scoreDiff',
#                 'expectedExtraScore',
#                 'adjExpectedScoreDiff',
#                 'yardsToGo',
#                 'yardsFromGoal',
#                 'offenseFavoritePoints',
#                 'secondsRemainingInPeriod',
#                 'remainingGameTime',
#                 'offenseScore',
#                 'remainingOffenseTOs',
#                 'remainingDefenseTOs',
#              ]
#
# cat_fields = [
#                 'eventtypeid',
#                 'period',
#                 'down',
#                 'playDesign'
#              ]

featureNames = [
                'score_diff',
                'adj_score_diff',
                'yards_to_go',
                'yards_from_goal',
                'offense_favorite_points',
                'seconds_remaining_in_period',
                'remaining_game_time',
                'offense_score',
                'remaining_offense_timeouts',
                'remaining_defense_timeouts',
                'event_type_id',
                'period',
                'down',
                'play_design',
                #'driveScoreDiffChange',
                'drive_outcome',
                'drive_start_score_diff'
                ]

num_fields_drive = [
                'score_diff',
                #'expectedExtraScore',
                #'adjExpectedScoreDiff',
                'yards_to_go',
                'yards_from_goal',
                'offense_favorite_points',
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
                'offense_favorite_points',
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

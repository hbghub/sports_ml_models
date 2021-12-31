import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

from src.CFB_regular_time_DL_model.utils import featureNames, TRAINING_END_SEASON

from src.utils import query_athena, save_dataset, \
                        calculate_score_diff, calculate_drive_outcome, calculate_remaining_timeout_cfb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def generate_features(win_df, training_end_season):

    featureNames.extend(['season', 'week', 'offense_team'])
    features = win_df[featureNames]

    #train-test split
    id_train = (win_df.season <= training_end_season).tolist()
    id_test  = (win_df.season > training_end_season).tolist()

    features_train = features[id_train]
    features_test  = features[id_test]

    label_train = win_df.offense_win[id_train]
    label_test  = win_df.offense_win[id_test]

    return features_train, features_test, label_train, label_test


def generate_dataset_local():
    print('start querying ... ')

    # (1) parse pbp data
    pbp_query_string = open("src/queries/pbp.sql", 'r').read()
    pbp_df = query_athena(pbp_query_string)
    print('pbp query is done!')

    # Fill in missing game time for regular period
    gd = pbp_df.groupby(['game_code', 'period'])
    pbp_df.seconds_remaining_in_period = gd['seconds_remaining_in_period'].fillna(method='ffill')

    gd = pbp_df.groupby(['game_code', 'period'])
    pbp_df.seconds_remaining_in_period = gd['seconds_remaining_in_period'].fillna(method='bfill')

    pbp_df['remaining_game_time'] = pbp_df.seconds_remaining_in_period + (4 - pbp_df.period).clip(0, ) * 900

    # overtime the clock is off, we may artificially assign a value of 1~2 min
    id = (pbp_df.seconds_remaining_in_period.isnull()) & (pbp_df.period > 4)
    pbp_df.loc[id, 'seconds_remaining_in_period'] = 120
    pbp_df.loc[id, 'remaining_game_time'] = 120

    # calculate scoreDiff
    calculate_score_diff(pbp_df)

    # calculate remaining TO, no sure if this works for CFB
    # > calculate timeout used in each half game
    calculate_remaining_timeout_cfb(pbp_df)

    # calculate outcome for each drive - applied for over-time model
    pbp_df = calculate_drive_outcome(pbp_df)

    # period start offense team: for OT modeling
    gb = pbp_df.groupby(['game_code', 'period']).head(1)[['game_code', 'period', 'offense_team']]
    gb.rename(columns={'offense_team': 'period_start_offense_team'}, inplace=True)
    pbp_df = pd.merge(pbp_df, gb, on=['game_code', 'period'], how='left')

    # NCAA not allowing for tie
    id = (pbp_df.home_team == pbp_df.offense_team) & (pbp_df.home_final_score > pbp_df.away_final_score)
    id = id | ((pbp_df.away_team == pbp_df.offense_team) & (pbp_df.home_final_score < pbp_df.away_final_score))

    pbp_df['offense_win'] = id.astype('bool')

    print(pbp_df.shape)

    features_train, features_test, label_train, label_test = generate_features(pbp_df, TRAINING_END_SEASON)

    save_dataset(features_train, "features_win_prob_CFB_reg_DL_train")

    save_dataset(features_test,  "features_win_prob_CFB_reg_DL_test")

    save_dataset(label_train, "label_win_prob_CFB_reg_DL_train")

    save_dataset(label_test,  "label_win_prob_CFB_reg_DL_test")

    return


if __name__ == '__main__':
    generate_dataset_local()
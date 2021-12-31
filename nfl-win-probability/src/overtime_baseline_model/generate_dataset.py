import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

from src.overtime_baseline_model.utils import featureNames

from src.utils import query_athena, save_dataset, TRAINING_END_SEASON, \
                        calculate_score_diff, calculate_drive_outcome, calculate_remaining_timeout


def generate_features(win_df, training_end_season):
    #featureNames.extend(['season', 'week'])
    features = win_df[featureNames]

    #train-test split
    id_train = (win_df.season <= training_end_season).tolist()
    id_test  = (win_df.season > training_end_season).tolist()

    features_train = features[id_train]
    features_test  = features[id_test]

    label_train = win_df.drive_outcome[id_train]
    label_test  = win_df.drive_outcome[id_test]

    return features_train, features_test, label_train, label_test


def generate_dataset_local():
    print('start querying ... ')

    # (1) parse pbp data
    pbp_query_string = open("src/queries/pbp_data.sql", 'r').read()
    pbp_df = query_athena(pbp_query_string)
    print('pbp query is done!')

    print(pbp_df.shape)

    # fix a DB error
    id = (pbp_df.game_code == 1744923) & (pbp_df.drive_id.isin([22]))

    pbp_df.loc[id, 'offense_team'] = 354
    pbp_df.loc[id, 'defense_team'] = 339

    # calculate score diff
    calculate_score_diff(pbp_df)

    # calculate remaining TO
    # > calculate timeout used in each half game
    calculate_remaining_timeout(pbp_df)

    # calculate outcome for each drive - applied for over-time model
    pbp_df = calculate_drive_outcome(pbp_df)

    # decide 'offense_win' field
    id = (pbp_df.home_team == pbp_df.offense_team) & (pbp_df.home_final_score > pbp_df.away_final_score)
    id = id | ((pbp_df.away_team == pbp_df.offense_team) & (pbp_df.home_final_score < pbp_df.away_final_score))

    pbp_df['offense_win'] = id.astype('bool')

    # (2) create pre-game odds
    pbp_query_string = open("src/queries/odds.sql", 'r').read()
    odds_df = query_athena(pbp_query_string)
    print('odds query is done!', odds_df.shape)

    win_df = pd.merge(pbp_df, odds_df, left_on=['game_code'], right_on=['game_code'])

    win_df['offense_favorite_points'] = win_df.favorite_points
    id = win_df.defense_team == win_df.favorite_team_id
    win_df.loc[id,'offense_favorite_points'] = win_df.loc[id,'favorite_points'] * (-1)

    features_train, features_test, label_train, label_test = generate_features(win_df, TRAINING_END_SEASON)

    save_dataset(features_train, "features_win_prob_ot_baseline_train")

    save_dataset(features_test,  "features_win_prob_ot_baseline_test")

    save_dataset(label_train, "label_win_prob_ot_baseline_train")

    save_dataset(label_test,  "label_win_prob_ot_baseline_test")

    #save_dataset(pbp_df, "pbp_df")

    return


if __name__ == '__main__':
    generate_dataset_local()
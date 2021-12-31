import pandas as pd
import numpy as np

from src.utils import query_athena, save_dataset, calculateRushingShareAdjByRank
from src.player_rush_share.utils import featureNames


def generate_features(game_df):
    # only keep cases that players are active and have override predictions
    # the filter of skipping starting weeks and zero output game is very important!!!

    #(~ game_df.exp_rushingShare.isnull()) & \
    id = (game_df.isActive) & (game_df.week > 1) & (
                game_df.rushertotalrushingattempts > 0)  # & (game_df.positionid==9)
    game_df = game_df[id]
    game_df.loc[:,'week'] = game_df.week.astype('float64')
    game_df.loc[:,'ytd_rushertotalrushingattempts'] = game_df.ytd_rushertotalrushingattempts.astype('float64')

    game_df.rename(columns={'ytd_rushingShareByPositionRank': 'ytd_rushingShareByPositionRankAdj'}, inplace=True)

    label_rush_share = game_df.rushingShare

    # game_df = game_df.reset_index()
    features = game_df[featureNames]
    features = pd.concat([features, game_df[['season', 'week', 'teamid', 'playerid']]], axis=1)

    id_train = (game_df.season <= 2018).tolist()
    id_test  = (game_df.season == 2019).tolist()

    # split data into train(2014~2018), test(2019)
    features_train = features[id_train]
    features_test = features[id_test]

    label_train = label_rush_share[id_train]
    label_test  = label_rush_share[id_test]

    print(features_train.shape, label_train.shape, features_test.shape, label_test.shape)

    return features_train, features_test, label_train, label_test


def generate_dataset_local():
    print('start querying ... ')

    #(1) game data
    query_string = open("src/queries/player_rush_game_stats.sql", 'r').read()
    game_df = query_athena(query_string)
    print('player_rush_game_stats query is done!')

    print(game_df.shape)

    # #(2) expected rates
    # query_string = open("src/queries/player_rush_expected_rates.sql", 'r').read()
    # exp_df = query_athena(query_string)
    # print('player_rush_expected_rates query is done!')
    #
    # game_df = pd.merge(game_df, exp_df, on=['gamecode', 'playerid'], how='left')

    # #(3) ytd data
    query_string = open("src/queries/player_rush_ytd_stats.sql", 'r').read()
    ytd_team_df = query_athena(query_string)
    print('player_rush_ytd_rates query is done!')

    game_df = pd.merge(game_df, ytd_team_df, on=['gamecode', 'teamid'], how='left')

    print(game_df.shape)

    # we should not simply fill the NA value since it may indicate missing the game
    id = game_df.rushertotalrushingattempts.isna()
    game_df['game_totalRushingAttempts'][id] = np.nan

    gd = game_df.groupby(['season', 'playerid'])
    breakpoint()
    game_df['ytd_totalRushingAttempts'] = gd.game_totalRushingAttempts.cumsum()
    game_df['ytd_rushertotalrushingattempts'] = gd.rushertotalrushingattempts.cumsum()

    game_df['ytd_rushingShare'] = game_df.ytd_rushertotalrushingattempts / game_df.ytd_totalRushingAttempts

    # For missing games, fill players' rushing share with previous game results
    game_df[['ytd_totalRushingAttempts', 'ytd_rushertotalrushingattempts', 'ytd_rushingShare']] = \
        gd[['ytd_totalRushingAttempts', 'ytd_rushertotalrushingattempts', 'ytd_rushingShare']].fillna(method='ffill')

    game_df[['ytd_totalRushingAttempts', 'ytd_rushertotalrushingattempts', 'ytd_rushingShare']] = \
        gd[['ytd_totalRushingAttempts', 'ytd_rushertotalrushingattempts', 'ytd_rushingShare']].shift(1)

    game_df[['game_totalRushingAttempts', 'rushertotalrushingattempts', 'rushingShare', 'ytd_rushingShare',
             'ytd_totalRushingAttempts', 'ytd_rushertotalrushingattempts']] = \
        game_df[['game_totalRushingAttempts', 'rushertotalrushingattempts', 'rushingShare', 'ytd_rushingShare',
                 'ytd_totalRushingAttempts', 'ytd_rushertotalrushingattempts']].fillna(0)

    #save_dataset(pbp_df, "pbp_df")

    # (4) previous (active) game rush share, regardless of team id

    gd = game_df.groupby(['season', 'playerid'])
    game_df['prev_rushingShare'] = gd.rushingShare.shift(1)
    game_df.info()
    game_df.prev_rushingShare = game_df.prev_rushingShare.fillna(0.0)

    # (5) create ytd targetShare by position rank
    # only ytd data is used, no baseline information is used!
    # 'gamecode' and 'playerid' are irrelevant here!

    ytd_byPosRank_df = game_df[['season', 'week', 'gamecode', 'teamid', 'playerid', 'positionid',
                                'rushertotalrushingattempts', 'totalrushingattempts', 'Rank']].copy()
    breakpoint()
    gd = ytd_byPosRank_df.groupby(['season', 'teamid', 'positionid', 'Rank'])

    ytd_byPosRank_df['ytd_rushingShareByPositionRank'] = gd.rushertotalrushingattempts.cumsum() / \
                                                         gd.totalrushingattempts.cumsum()
    ytd_byPosRank_df['ytd_rushingShareByPositionRank'] = gd.ytd_rushingShareByPositionRank.shift(1)

    # 6
    re = calculateRushingShareAdjByRank(game_df, positionIds=[9], seasons=[2018, 2019],
                                        ytd_byPosRank_df=ytd_byPosRank_df, printDetails=False)

    adjustedRates = pd.concat(re, ignore_index=True)

    game_df = pd.merge(game_df,
                     adjustedRates[['teamid','gamecode','playerid','onFieldRank','ytd_rushingShareByPositionRank']],
                     on=['teamid','gamecode','playerid'],
                     how='left')

    # we create a new column 'ytd_targetShareAdj' to contain ytd data with adjustment by injury situation
    game_df['ytd_rushingShare_2'] = game_df.ytd_rushingShare

    # skip for now
    # id = game_df.ytd_rushingShareByPositionRank.isnull()
    # game_df.ytd_rushingShare_2[~id] = game_df.ytd_rushingShareByPositionRank[~id]

    # (7) create baseline case from ytd data
    # Note: each year, many new players join the league without baseline-info;
    #       many players retire so their baseline won't be in use for next year

    baseline_df = game_df[['season', 'playerid',
                           'ytd_totalRushingAttempts',
                           'ytd_rushertotalrushingattempts',
                           'ytd_rushingShare',
                           'ytd_rushingShare_2']].copy()
    baseline_df = baseline_df.groupby(['season', 'playerid']).tail(1)

    baseline_df.rename(columns={
        'ytd_totalRushingAttempts': 'base_totalRushingAttempts',
        'ytd_rushertotalrushingattempts': 'base_rushertotalrushingattempts',
        'ytd_rushingShare': 'base_rushingShare',
        'ytd_rushingShare_2': 'base_rushingShareAdj'},
        inplace=True)

    baseline_df.season = baseline_df.season + 1

    # merge baseline info into game_df, in this case, we will lose 2017

    game_df = pd.merge(game_df, baseline_df, on=['season', 'playerid'], how='left')

    id = game_df.season.isin([2018, 2019])
    game_df = game_df[id]

    game_df[['base_rushingShare', 'base_rushingShareAdj', 'base_totalRushingAttempts', 'base_rushertotalrushingattempts']] = \
        game_df[['base_rushingShare', 'base_rushingShareAdj', 'base_totalRushingAttempts',
                 'base_rushertotalrushingattempts']].fillna(0)

    # 8
    # Weighted historical values
    calculate_player_adjusted_rush_share(alpha=5.0, game_df=game_df)

    # 9 create rush share by position id
    gd = game_df.groupby(['season', 'teamid', 'positionid', 'week'])
    tmp = gd.rushertotalrushingattempts.sum() / gd.totalrushingattempts.median()
    tmp.rename('teamPositionRushShare', inplace=True)
    tmp = tmp.to_frame()  # .reset_index()

    gd = tmp.groupby(level=['season', 'teamid', 'positionid'], as_index=False, group_keys=False)
    tmp2 = gd.expanding().mean()
    tmp2.rename(columns={'teamPositionRushShare': 'm_teamPositionRushShare'}, inplace=True)

    gd = tmp2.groupby(level=['season', 'teamid', 'positionid'], as_index=False, group_keys=False)
    tmp2 = gd.shift(1)
    tmp['m_teamPositionRushShare'] = tmp2['m_teamPositionRushShare']

    tmp.reset_index()

    game_df = pd.merge(game_df, tmp, on=['season', 'teamid', 'week', 'positionid'], how='left')

    # normalized w_rushShareAdj, only for RB
    game_df['ytd_rushingShareAdj_norm'] = game_df.ytd_rushingShareAdj

    id = game_df.positionid == "RB" #9

    # skip normalization for now
    game_df.loc[id, 'ytd_rushingShareAdj_norm'] = game_df[id].teamPositionRushShare / game_df[
         id].m_teamPositionRushShare * game_df[id].ytd_rushingShareAdj

    features_train, features_test, label_train, label_test = generate_features(game_df)

    save_dataset(features_train, "features_rush_share_train")

    save_dataset(features_test, "features_rush_share_test")

    save_dataset(label_train, "label_rush_share_train")

    save_dataset(label_test, "label_rush_share_test")

def calculate_player_adjusted_rush_share(alpha, game_df):
    w = game_df.ytd_totalRushingAttempts * alpha / \
        (game_df.ytd_totalRushingAttempts * alpha + game_df.base_totalRushingAttempts)
    id = (game_df.ytd_totalRushingAttempts == 0) & (game_df.base_totalRushingAttempts == 0)
    w[id] = 1.0

    game_df['ytd_rushingShareAdj'] = game_df.ytd_rushingShare_2 * w + game_df.base_rushingShareAdj * (1 - w)
    return

if __name__ == '__main__':
    generate_dataset_local()
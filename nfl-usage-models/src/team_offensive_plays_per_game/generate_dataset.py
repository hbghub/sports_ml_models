import pandas as pd
import numpy as np

import src.utils as utils

from src.utils import query_athena, save_dataset
from src.team_offensive_plays_per_game.utils import featureNames


def generate_features(game_df):
    # (1)
    # alpha controls the weighting factor calculation
    alpha = 2.0

    w = game_df.ytd_gameTime * alpha / (game_df.ytd_gameTime * alpha + game_df.base_gameTime)

    game_df['ytd_offensivePlaysPerGameAdj'] = game_df.ytd_offensivePlaysPerGame * w + game_df.base_offensivePlaysPerGame * (
                1 - w)

    game_df['ytd_passPlaysPerGameAdj'] = game_df.ytd_passPlaysPerGame * w + game_df.base_passPlaysPerGame * (1 - w)
    game_df['ytd_passRatioAdj'] = game_df['ytd_passPlaysPerGameAdj'] / game_df['ytd_offensivePlaysPerGameAdj']

    w_gameTime = (game_df.ytd_TOPperGame + game_df.ytd_o_TOPperGame) * w + \
                 (game_df.base_TOPperGame + game_df.base_o_TOPperGame) * (1 - w)

    # do we need this game time adjustment????
    game_df['ytd_offensivePlaysAdj'] = 3600 / w_gameTime * game_df['ytd_offensivePlaysPerGameAdj']

    # (2)
    # team pace adjustment
    w_pace = game_df.ytd_pace * w + game_df.base_pace * (1 - w)
    w_pace_conceded = game_df.ytd_pace_conceded * w + game_df.base_pace_conceded * (1 - w)

    w_o_pace = game_df.ytd_o_pace * w + game_df.base_o_pace * (1 - w)
    w_o_pace_conceded = game_df.ytd_o_pace_conceded * w + game_df.base_o_pace_conceded * (1 - w)

    # adjustment term by pace

    game_df['ytd_paceConcededAdj'] = w_o_pace - w_pace_conceded

    game_df['ytd_paceAdj'] = w_pace - w_o_pace_conceded

    # (3) match-up historical TPPP

    game_df['ytd_passingYardsAdj'] = game_df.ytd_passingYardsPerAttempt - game_df.ytd_o_passingYardsPerAttempt
    game_df['ytd_rushingYardsAdj'] = game_df.ytd_rushingYardsPerAttempt - game_df.ytd_o_rushingYardsPerAttempt

    game_df['ytd_passingYardsAdj2'] = game_df.ytd_passingYardsPerAttempt - game_df.ytd_o_passingYardsPerAttempt_conceded
    game_df['ytd_rushingYardsAdj2'] = game_df.ytd_rushingYardsPerAttempt - game_df.ytd_o_rushingYardsPerAttempt_conceded

    # (4)
    # for total plays
    features_TPPG = game_df[featureNames]
    
    id = (~features_TPPG.isna().any(axis=1)).tolist()
    features = features_TPPG[id]
    print(features.shape, game_df.shape)

    label_TPPG = game_df[id].totaloffensiveplays.astype(float)

    id_train = (game_df.season[id] <= 2018).tolist()
    id_test = (game_df.season[id] == 2019).tolist()
    print(len(id_train), len(id_test))

    game_df = game_df[id] #.reset_index()

    features = pd.concat([features, game_df[['season', 'week', 'teamid']]], axis=1)

    # split data into train(2014~2018), test(2019)
    features_train = features.loc[id_train]
    features_test = features.loc[id_test]

    label_train = label_TPPG[id_train]
    label_test  = label_TPPG[id_test]

    print(features_train.shape, label_train.shape, features_test.shape, label_test.shape)

    return features_train, features_test, label_train, label_test


def generate_dataset_local():
    print('start querying ... ')

    #(1) game data
    query_string = open("src/queries/team_game_stats.sql", 'r').read()
    game_df = query_athena(query_string)
    print('team_game_stats query is done!')

    #(2) expected rates
    query_string = open("src/queries/team_expected_rates.sql", 'r').read()
    exp_df = query_athena(query_string)
    print('team_expected_rates query is done!')

    game_df = pd.merge(game_df, exp_df, on=['season', 'teamid', 'week'], how='inner')

    #(3) ytd data
    query_string = open("src/queries/team_ytd_rates.sql", 'r').read()
    ytd_df = query_athena(query_string)
    print('team_ytd_rates query is done!')

    game_df = pd.merge(game_df, ytd_df, on=['season', 'teamid', 'week'], how='left')

    query_string = open("src/queries/team_ytd_rates_conceded.sql", 'r').read()
    ytd_df_conceded = query_athena(query_string)
    print('team_ytd_rates_conceded query is done!')

    game_df = pd.merge(game_df, ytd_df_conceded, on=['season', 'teamid', 'week'], how='left')

    ## add opponent ytd_data
    tmp = game_df[['season', 'teamid', 'week', 'ytd_pace', 'ytd_pace_conceded', 'ytd_offensivePlaysPerGame']].copy()
    tmp.rename(columns={'teamid': 'opponentteamid',
                        'ytd_pace': 'ytd_o_pace',
                        'ytd_pace_conceded': 'ytd_o_pace_conceded',
                        'ytd_offensivePlaysPerGame': 'ytd_o_offensivePlaysPerGame'}, inplace=True)
    game_df = pd.merge(game_df, tmp, on=['season', 'opponentteamid', 'week'], how='left')

    ##(3.1) ytd_yards
    query_string = open("src/queries/team_stats_game.sql", 'r').read()
    yards_df = query_athena(query_string)
    print('team_ytd_rates_conceded query is done!')

    yards_df = pd.merge(yards_df, game_df[['season', 'teamid', 'gamecode', 'opponentteamid']],
                        on=['season', 'teamid', 'gamecode'], how='right')

    yards_df = utils.calculate_team_ytd_passing_yards(yards_df)

    game_df = pd.merge(game_df,
                       yards_df[['season', 'teamid', 'week',
                                 'ytd_passingYardsPerAttempt', 'ytd_rushingYardsPerAttempt',
                                 'ytd_passingYardsPerAttempt_conceded', 'ytd_rushingYardsPerAttempt_conceded',
                                 'ytd_o_passingYardsPerAttempt', 'ytd_o_rushingYardsPerAttempt',
                                 'ytd_o_passingYardsPerAttempt_conceded', 'ytd_o_rushingYardsPerAttempt_conceded']],
                       on=['season', 'teamid', 'week'], how='left')

    ##(3.2) ytd_scrambles
    query_string = open("src/queries/team_scramble.sql", 'r').read()
    scrambles_df = query_athena(query_string)
    print('team_scrambles query is done!')

    utils.calculate_team_ytd_scrambles(scrambles_df)

    # merge scrambles into game_df
    game_df = pd.merge(game_df, scrambles_df[['season', 'teamid', 'week', 'ytd_scrambles']],
                       on=['season', 'teamid', 'week'], how='left')

    game_df['ytd_scrambleRatio'] = game_df.ytd_scrambles / game_df.ytd_passPlaysPerGame


    #(4) create baseline case from ytd data
    baseline_df = game_df[['season', 'teamid', 'ytd_gameTime', 'ytd_pace', 'ytd_pace_conceded',
                           'ytd_passingpercentage', 'ytd_offensivePlaysPerGame', 'ytd_passPlaysPerGame',
                           'ytd_TOPperGame']]
    baseline_df = baseline_df.groupby(['season', 'teamid']).tail(1)

    baseline_df.rename(columns={'ytd_gameTime': 'base_gameTime',
                                'ytd_pace': 'base_pace',
                                'ytd_pace_conceded': 'base_pace_conceded',
                                'ytd_offensivePlaysPerGame': 'base_offensivePlaysPerGame',
                                'ytd_passPlaysPerGame': 'base_passPlaysPerGame',
                                'ytd_passingpercentage': 'base_passingpercentage',
                                'ytd_TOPperGame': 'base_TOPperGame'},
                       inplace=True)

    baseline_df.loc[:, 'season'] = baseline_df.season + 1

    # merge baseline info into game_df, in this case, we will lose 2013
    game_df = pd.merge(game_df, baseline_df, on=['season', 'teamid'], how='inner')

    #(5) compute regressed YTD team passing %, using ytd and baseline info
    # compute expected passing play
    utils.calculate_team_expected_passing(game_df)

    #(6) add baseline info for opponent team
    tmp = game_df[['teamid', 'gamecode', 'ytd_TOPperGame', 'ytd_totalPointsPerGame', 'ytd_totalSacksPerGame',
                   'base_pace', 'base_pace_conceded', 'base_offensivePlaysPerGame', 'base_TOPperGame']].copy()
    tmp.rename(columns={'teamid': 'opponentteamid',
                        'ytd_TOPperGame': 'ytd_o_TOPperGame',
                        'ytd_totalPointsPerGame': 'ytd_o_totalPointsPerGame',
                        'ytd_totalSacksPerGame': 'ytd_o_totalSacksPerGame',
                        'base_pace': 'base_o_pace',
                        'base_pace_conceded': 'base_o_pace_conceded',
                        'base_offensivePlaysPerGame': 'base_o_offensivePlaysPerGame',
                        'base_passingpercentage': 'base_o_passingpercentage',
                        'base_TOPperGame': 'base_o_TOPperGame'},
               inplace=True)

    game_df = pd.merge(game_df, tmp, on=['opponentteamid', 'gamecode'], how='left')

    #(7) pre-game odds
    query_string = open("src/queries/team_pregame_odds.sql", 'r').read()
    odds_df = query_athena(query_string)
    print('team_pregame_odds query is done!')

    # merge odds into game_df
    tmp = game_df.loc[game_df.season >= 2016, ['gamecode', 'teamid', 'opponentteamid']].copy()

    tmp2 = odds_df[['gamecode', 'favoriteTeamId', 'homeMoneyDecimal', 'awayMoneyDecimal', 'favoritePoints']].copy()
    #tmp2.loc[:, 'odds'] = odds_df.homeMoneyDecimal / odds_df.awayMoneyDecimal
    tmp2['odds'] = odds_df.homeMoneyDecimal / odds_df.awayMoneyDecimal

    tmp2 = pd.merge(tmp, tmp2[['gamecode', 'favoriteTeamId', 'odds', 'favoritePoints']], on=['gamecode'], how='left')

    id = tmp2.teamid == tmp2.favoriteTeamId
    tmp2.loc[id, 'odds'] = 1 / tmp2.odds[id]

    tmp2.loc[id, 'favoritePoints'] = abs(tmp2.favoritePoints[id])
    tmp2.loc[~id, 'favoritePoints'] = -abs(tmp2.favoritePoints[~id])

    # merge odds df into game_df
    game_df = pd.merge(game_df, tmp2[['gamecode', 'teamid', 'odds', 'favoritePoints']], on=['gamecode', 'teamid'],
                       how='left')
    print(game_df.shape)

    features_train, features_test, label_train, label_test = generate_features(game_df)

    save_dataset(features_train, "features_TPPG_train")

    save_dataset(features_test,  "features_TPPG_test")

    save_dataset(label_train, "label_TPPG_train")

    save_dataset(label_test,  "label_TPPG_test")

    #save_dataset(pbp_df, "pbp_df")

if __name__ == '__main__':
    generate_dataset_local()
import pandas as pd
import numpy as np

from src.utils import query_athena, save_dataset
from src.player_target_share.utils import featureNames


def generate_features(game_df):
    game_df.ytd_rank = game_df.ytd_rank.astype('float64')

    game_df.rename(columns={'ytd_targetShareByPositionRank':'ytd_targetShareByPositionRankAdj'}, inplace=True)

    label_target_share = game_df.targetShare

    features = game_df[featureNames]

    print("Features:", featureNames)

    features = pd.concat([features, game_df[['season', 'week', 'teamid', 'playerid']]], axis=1)

    id_train = (game_df.season <= 2018).tolist()
    id_test  = (game_df.season == 2019).tolist()

    # split data into train(2014~2018), test(2019)
    features_train = features[id_train]
    features_test  = features[id_test]

    label_train = label_target_share[id_train]
    label_test  = label_target_share[id_test]

    print(features_train.shape, label_train.shape, features_test.shape, label_test.shape)

    return features_train, features_test, label_train, label_test


# Function to calculate target share by ranks for each positin type when there is a roster change
# return a list of objects, each object contains targetShares for players for a game when roster change happens
# Note: there is a bias to use targetShare by position to estimate each player's performance!!!
def calculateTargetShareAdjByRank(game_df, positionIds, seasons, rank_df, printDetails=False):
    adjustedRates = []

    teams = game_df.teamid.unique()

    for season in seasons:
        for team in teams:
            for positionId in positionIds:
                # print(team)

                id = (game_df.season == season) & (game_df.teamid == team) & (game_df.positionid == positionId)
                one_team = game_df[id]

                # print(one_team.shape)

                ranking_data = []

                for i, week in enumerate(one_team.week.unique()):
                    id = one_team.week == week

                    if i == 0:
                        ranking_data = one_team.loc[id, ['receivertotaltargetsontruepassattempts', 'playerid']].copy()
                        ranking_data['ytd_rank'] = -1
                        ranking_data.set_index('playerid', inplace=True)
                        continue

                    data = one_team.loc[id, ['teamid', 'gamecode', 'positionid', 'playerid',
                                             'receivertotaltargetsontruepassattempts']].copy()
                    data.set_index('playerid', inplace=True)

                    # add according to playerid index
                    ranking_data = ranking_data.add(data, fill_value=0)
                    ranking_data.sort_values('receivertotaltargetsontruepassattempts', inplace=True, ascending=False)
                    ranking_data.loc[:, 'ytd_rank'] = np.arange(len(ranking_data)) + 1

                    current_week_data = one_team[one_team.week == week]

                    # check if any top (2) players is missing for this week
                    missingPlayers = [player for player in ranking_data.index.values
                                      if player not in current_week_data.playerid.values and
                                      ranking_data.loc[player].ytd_rank <= 2]

                    if missingPlayers:
                        if printDetails:
                            for player in missingPlayers:
                                print(week, player, ranking_data.loc[player].ytd_rank)

                        # re-arrange ranks of active players to reflect currently predicted rank
                        data['onFieldRank'] = ranking_data.loc[data.index].ytd_rank. \
                            rank(method='first', na_option='bottom')
                        data = data.astype({'onFieldRank': 'int64'})
                        data.reset_index(inplace=True)

                        # merge target%_by_rank into data
                        data = pd.merge(data,
                                        rank_df[['teamid', 'gamecode', 'positionid', 'Rank',
                                                 'ytd_targetShareByPositionRank']],
                                        left_on=['teamid', 'gamecode', 'positionid', 'onFieldRank'],
                                        right_on=['teamid', 'gamecode', 'positionid', 'Rank'], how='left')

                        # adjustment
                        # data.ytd_targetShareByPositionRank = data.ytd_targetShareByPositionRank * 0.9

                        adjustedRates.append(data)

    return (adjustedRates)


def generate_dataset_local():
    print('start querying ... ')

    #(1) game data
    query_string = open("src/queries/player_game_stats.sql", 'r').read()
    game_df = query_athena(query_string)
    print('player_game_stats query is done!')

    # #(2) expected rates, may not need this step!!!
    # query_string = open("src/queries/player_expected_rates.sql", 'r').read()
    # exp_df = query_athena(query_string)
    # print('player_expected_rates query is done!')
    #
    # game_df = pd.merge(game_df, exp_df, on=['gamecode', 'playerid'], how='left')

    # #(3) ytd data
    query_string = open("src/queries/player_ytd_stats.sql", 'r').read()
    ytd_df = query_athena(query_string)
    print('player_ytd_rates query is done!')

    # prepare ytd_truepassattempts for each player. in this way, no update for missed games
    gd = ytd_df.groupby(['season', 'playerid'])

    ytd_df['ytd_totaltruepassattempts'] = gd.game_totaltruepassattempts.cumsum()
    ytd_df['ytd_totaltruepassattempts'] = gd.ytd_totaltruepassattempts.shift(1)

    ytd_df['ytd_targetShare'] = ytd_df.ytd_totalTargetsOnTruePassAttempts / ytd_df.ytd_totaltruepassattempts

    game_df = pd.merge(game_df, ytd_df[['gamecode', 'playerid',
                                        'ytd_rank',
                                        'ytd_onFieldTotalTruePassAttempts',
                                        'ytd_totalTargetsOnTruePassAttempts',
                                        'ytd_totaltruepassattempts',
                                        'ytd_targetShare']],
                       on=['gamecode', 'playerid'], how='left')

    print(game_df.shape)

    rank_df = game_df[['season', 'week', 'gamecode', 'teamid', 'playerid', 'positionid',
                       'receivertotaltargetsontruepassattempts', 'totaltruepassattempts', 'Rank']].copy()

    gd = rank_df.groupby(['season', 'teamid', 'positionid', 'Rank'])

    rank_df['ytd_targetShareByPositionRank'] = gd.receivertotaltargetsontruepassattempts.cumsum() / \
                                               gd.totaltruepassattempts.cumsum()
    rank_df['ytd_targetShareByPositionRank'] = gd.ytd_targetShareByPositionRank.shift(1)

    re = calculateTargetShareAdjByRank(game_df, positionIds=[1, 7, 9], seasons=[2017, 2018, 2019], rank_df=rank_df)

    adjustedRates = pd.concat(re, ignore_index=True)

    game_df = pd.merge(game_df,
                       adjustedRates[
                           ['teamid', 'gamecode', 'playerid', 'onFieldRank', 'ytd_targetShareByPositionRank']],
                       on=['teamid', 'gamecode', 'playerid'],
                       how='left')
    print(game_df.shape)

    # we create a new column 'ytd_targetShareAdj' column to contain ytd data with adjustment by injury situation
    game_df['ytd_targetShare_2'] = game_df.ytd_targetShare

#    id = game_df.ytd_targetShareByPositionRank.isnull()
#    game_df.ytd_targetShare_2[~id] = game_df.ytd_targetShareByPositionRank[~id]

    print(game_df.shape)

    # create baseline case from ytd data
    baseline_df = game_df[['season', 'playerid', 'ytd_onFieldTotalTruePassAttempts',
                           'ytd_totalTargetsOnTruePassAttempts', 'ytd_totaltruepassattempts',
                           'ytd_targetShare', 'ytd_targetShare_2']].copy()
    baseline_df = baseline_df.groupby(['season', 'playerid']).tail(1)

    baseline_df.rename(columns={'ytd_onFieldTotalTruePassAttempts': 'base_onFieldTotalTruePassAttempts',
                                'ytd_totalTargetsOnTruePassAttempts': 'base_totalTargetsOnTruePassAttempts',
                                'ytd_totaltruepassattempts': 'base_totaltruepassattempts',
                                'ytd_targetShare': 'base_targetShare',
                                'ytd_targetShare_2': 'base_targetShareAdj'},
                       inplace=True)

    baseline_df.season = baseline_df.season + 1

    # merge baseline info into game_df, in this case, we will lose 2017

    game_df = pd.merge(game_df, baseline_df, on=['season', 'playerid'], how='left')

    print(game_df.shape)

    id = game_df.season.isin([2018, 2019])
    game_df = game_df[id]

    game_df.ytd_targetShareByPositionRank.fillna(-1, inplace=True)

    game_df.fillna(0, inplace=True)

    # Weighted historical values
    # note the fill of na is after the target percentage has been calculated
    calculate_player_adjusted_target_share(5.0, game_df)

    print(game_df.shape)

    features_train, features_test, label_train, label_test = generate_features(game_df)

    save_dataset(features_train, "features_target_share_train")

    save_dataset(features_test,  "features_target_share_test")

    save_dataset(label_train, "label_target_share_train")

    save_dataset(label_test,  "label_target_share_test")

    #save_dataset(pbp_df, "pbp_df")


def calculate_player_adjusted_target_share(alpha, game_df):
    w = game_df.ytd_onFieldTotalTruePassAttempts * alpha / \
        (game_df.ytd_onFieldTotalTruePassAttempts * alpha + game_df.base_onFieldTotalTruePassAttempts)
    id = (game_df.ytd_onFieldTotalTruePassAttempts == 0) & (game_df.base_onFieldTotalTruePassAttempts == 0)
    w[id] = 1.0
    game_df['ytd_targetShareAdj'] = game_df.ytd_targetShare_2 * w + game_df.base_targetShare * (1 - w)
    return


if __name__ == '__main__':
    generate_dataset_local()
import pandas as pd
import numpy as np
import os
import mlflow

from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from pyathena import connect
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

import stats.predeval as predeval

def query_athena(query_string):
    conn = connect(
        s3_staging_dir='s3://aws-athena-query-results-323906537337-us-east-1/',
        region_name='us-east-1'
    )
    return pd.read_sql(query_string, conn)


def local_data_path():
    return os.path.abspath('runtime/datasets')


def save_dataset(df, name):
    if not os.path.exists(local_data_path()):
        os.makedirs(local_data_path())
    df.to_csv(os.path.join(local_data_path(), name + '.csv'), index=False)


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')


def model_train(model_hyperparams, model_version, features, labels, transform_pipeline):
    if model_version in ['0.0.01']:
        model = LogisticRegression(**model_hyperparams)
    elif model_version in ['0.0.02']:
        model = LinearSVC(**model_hyperparams)
    elif model_version in ['0.0.03']:
        model = RandomForestClassifier(**model_hyperparams)
    elif model_version in ['0.0.04']:
        model = RandomForestRegressor(**model_hyperparams)
    elif model_version in ['0.0.05']:
        model = SGDRegressor(**model_hyperparams)
    elif model_version in ['0.0.06']:
        model = LinearSVR(**model_hyperparams)
    elif model_version in ['0.0.07']:
        model = LinearRegression(**model_hyperparams)
    else:
        print("Error: wrong MODEL_VERSION:", model_version)

    pipe = Pipeline([('columnTransfer', transform_pipeline), ('model', model)])

    pipe.fit(features, labels)

    return pipe


# feature_train, label_train, feature_test, label_test: string represent file name
def evaluate_classification(model, features_train, label_train, features_test, label_test):
    if mlflow.active_run() is None:
        raise Exception('Cannot call evaluate() method without first setting an active mlflow run.')

    train_features = load_dataset(features_train)
    train_labels = load_dataset(label_train).iloc[:, 0]
    test_features = load_dataset(features_test)
    test_labels = load_dataset(label_test).iloc[:, 0]

    train_probs = pd.Series(model.predict_proba(train_features)[:, 1])
    test_probs = pd.Series(model.predict_proba(test_features)[:, 1])
    train_preds = pd.Series(model.predict(train_features))
    test_preds = pd.Series(model.predict(test_features))

    train_roc_auc = roc_auc_score(train_labels, train_probs)
    test_roc_auc  = roc_auc_score(test_labels, test_probs)
    mlflow.log_metric('train_roc_auc', train_roc_auc)
    mlflow.log_metric('test_roc_auc',  test_roc_auc)
    print(train_roc_auc, test_roc_auc)

    #predeval.plot_calibration_curve(test_labels, test_probs)

    return


def evaluate_regression(model, features_train, label_train, features_test, label_test):
    if mlflow.active_run() is None:
        raise Exception('Cannot call evaluate() method without first setting an active mlflow run.')

    train_preds = pd.Series(model.predict(features_train))
    test_preds = pd.Series(model.predict(features_test))

    re_in = predeval.classical_regression_metrics(label_train, train_preds)
    print(re_in)

    re_out = predeval.classical_regression_metrics(label_test, test_preds)
    print(re_out)

    mlflow.log_metric('train_r2', re_in['r2'])
    mlflow.log_metric('test_r2',  re_out['r2'])

    mlflow.log_metric('train_rmse', re_in['rmse'])
    mlflow.log_metric('test_rmse',  re_out['rmse'])

    mlflow.log_metric('train_mae', re_in['mae'])
    mlflow.log_metric('test_mae',  re_out['mae'])

    print("R2: {:.2%}, {:.2%}".format(re_in['r2'], re_out['r2']))
    print("rmse: {:.2f}, {:.2f}".format(re_in['rmse'], re_out['rmse']) )
    print("mae: {:.2f}, {:.2f}".format(re_in['mae'], re_out['mae']) )

    return


# Function to calculate target share by ranks for each positin type when there is a roster change
# return a list of objects, each object contains targetShares for players for a game when roster change happens
# Note: there is a bias to use targetShare by position to estimate each player's performance!!!

# For rush share, we may have to rely on previous game ranking instead of ytd_ranking!

def calculateRushingShareAdjByRank(game_df, positionIds, seasons, ytd_byPosRank_df, printDetails=False):
    adjustedRates = []

    teams = game_df.teamid.unique()

    for season in seasons:
        for team in teams:
            for positionId in positionIds:
                print(team)

                id = (game_df.season == season) & (game_df.teamid == team) & (game_df.positionid == positionId)
                one_team = game_df[id].copy()

                # ranking_data is used to dynamically track ytd players' ranking for certain position
                ranking_data = []

                for i, week in enumerate(one_team.week.unique()):
                    id = (one_team.week == week) & (one_team.isActive)

                    if i == 0:
                        ranking_data = one_team.loc[id, ['rushertotalrushingattempts', 'playerid']].copy()
                        ranking_data['ytd_rank'] = -1
                        ranking_data.set_index('playerid', inplace=True)
                        continue

                    data = one_team.loc[id, ['teamid', 'gamecode', 'positionid', 'playerid',
                                             'rushertotalrushingattempts', 'rushingShare', 'ytd_rushingShare']].copy()
                    data.set_index('playerid', inplace=True)

                    # add according to playerid index
                    ranking_data = ranking_data.add(data[['rushertotalrushingattempts']], fill_value=0)
                    ranking_data.sort_values('rushertotalrushingattempts', inplace=True, ascending=False)
                    ranking_data.loc[:, 'ytd_rank'] = np.arange(len(ranking_data)) + 1

                    # print(week, ranking_data)

                    current_week_data = one_team[id]
                    activeMajorPlayers = current_week_data.playerid[current_week_data.rushertotalrushingattempts > 0]

                    # check if any top (1) player(s) is missing for this week
                    missingPlayers = [player for player in ranking_data.index.values
                                      # if player not in current_week_data.playerid.values and
                                      if player not in activeMajorPlayers.values and
                                      ranking_data.loc[player].ytd_rank <= 2]

                    # print(current_week_data[['week','playerid','rushertotalrushingattempts','isActive']])

                    if missingPlayers:
                        if printDetails:
                            for player in missingPlayers:
                                print(week, player, ranking_data.loc[player].ytd_rank)
                                print(ranking_data)

                        # re-arrange ranks of active players to reflect currently predicted rank
                        data['onFieldRank'] = ranking_data.loc[data.index].ytd_rank. \
                            rank(method='first', na_option='bottom')
                        data = data.astype({'onFieldRank': 'int64'})
                        data.reset_index(inplace=True)

                        # merge target%_by_rank into data
                        data = pd.merge(data,
                                        ytd_byPosRank_df[['teamid', 'gamecode', 'positionid', 'Rank',
                                                          'ytd_rushingShareByPositionRank']],
                                        left_on=['teamid', 'gamecode', 'positionid', 'onFieldRank'],
                                        right_on=['teamid', 'gamecode', 'positionid', 'Rank'], how='left')

                        # adjustment
                        # data.ytd_targetShareByPositionRank = data.ytd_targetShareByPositionRank * 0.9

                        adjustedRates.append(data)

    return (adjustedRates)


def calculate_team_expected_passing(game_df):
    game_df['reg_passingpercentage'] = (350 * game_df.base_passingpercentage +
                                        game_df.ytd_totalPlays * game_df.ytd_passingpercentage) / \
                                       (350 + game_df.ytd_totalPlays)
    game_df['exp_passingPlays'] = game_df.exp_totalPlays * game_df.reg_passingpercentage


def calculate_team_ytd_scrambles(scrambles_df):
    # calculate cumulated sum and then yardsPerAttempt
    gd = scrambles_df.groupby(['season', 'teamid'])
    scrambles_df['ytd_scrambles'] = gd.game_scrambles.cumsum() / gd.idx.cumsum()
    scrambles_df.loc[:, 'ytd_scrambles'] = gd.ytd_scrambles.shift(1)


def calculate_team_ytd_passing_yards(yards_df):
    # calculate cumulated sum and then yardsPerAttempt
    gd = yards_df.groupby(['season', 'teamid'])
    yards_df['ytd_passingYardsPerAttempt'] = gd.passingYards.cumsum() / gd.passingAttempts.cumsum()
    yards_df['ytd_rushingYardsPerAttempt'] = gd.rushingYards.cumsum() / gd.rushingAttempts.cumsum()
    yards_df.loc[:, 'ytd_passingYardsPerAttempt'] = gd.ytd_passingYardsPerAttempt.shift(1)
    yards_df.loc[:, 'ytd_rushingYardsPerAttempt'] = gd.ytd_rushingYardsPerAttempt.shift(1)

    # calculate opponent cumulated sum and then yardsPerAttempt
    yards_df.sort_values(['season', 'opponentteamid', 'week'], inplace=True)
    gd = yards_df.groupby(['season', 'opponentteamid'])
    yards_df['ytd_o_passingYardsPerAttempt_conceded'] = gd.passingYards.cumsum() / gd.passingAttempts.cumsum()
    yards_df['ytd_o_rushingYardsPerAttempt_conceded'] = gd.rushingYards.cumsum() / gd.rushingAttempts.cumsum()
    yards_df.loc[:, 'ytd_o_passingYardsPerAttempt_conceded'] = gd.ytd_o_passingYardsPerAttempt_conceded.shift(1)
    yards_df.loc[:, 'ytd_o_rushingYardsPerAttempt_conceded'] = gd.ytd_o_rushingYardsPerAttempt_conceded.shift(1)

    # merge to get team conceded yards
    tmp = yards_df[['season', 'opponentteamid', 'week',
                    'ytd_o_passingYardsPerAttempt_conceded', 'ytd_o_rushingYardsPerAttempt_conceded']].copy()
    tmp.rename(columns={'opponentteamid': 'teamid',
                        'ytd_o_passingYardsPerAttempt_conceded': 'ytd_passingYardsPerAttempt_conceded',
                        'ytd_o_rushingYardsPerAttempt_conceded': 'ytd_rushingYardsPerAttempt_conceded'}, inplace=True)
    yards_df = pd.merge(yards_df, tmp, on=['season', 'teamid', 'week'], how='left')

    # create opponents passing/rushing yards
    tmp = yards_df[['season', 'teamid', 'week', 'ytd_passingYardsPerAttempt', 'ytd_rushingYardsPerAttempt']].copy()
    tmp.rename(columns={'teamid': 'opponentteamid',
                        'ytd_passingYardsPerAttempt': 'ytd_o_passingYardsPerAttempt',
                        'ytd_rushingYardsPerAttempt': 'ytd_o_rushingYardsPerAttempt'}, inplace=True)
    yards_df = pd.merge(yards_df, tmp, on=['season', 'week', 'opponentteamid'], how='left')

    yards_df.sort_values(by=['season','teamid','week'], axis=0, inplace=True)

    return yards_df


from src.utils import query_athena, team_dict, num_fields_player, cat_fields_player, save_dataset, \
                        TESTING_SEASON, feature_names

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def generate_features(player_df, testing_season):
    # data type conversion
    player_df = player_df.astype(
        {'minutes': 'float64', 'points': 'float64', 'fg_attempt': 'float64', 'fg_made': 'float64',
         'ft_attempt': 'float64', 'ft_made': 'float64', 'point_3_attempt': 'float64', 'point_3_made': 'float64',
         'offensive_rebounds': 'float64', 'defensive_rebounds': 'float64', 'assists': 'float64', 'blocks': 'float64',
         'turnovers': 'float64'})

    player_df = player_df.astype({'season': 'int64', 'player_id': 'int64', 'game_time': 'int64', 'game_code': 'int64',
                                  'team_id': 'int64', 'at_home': 'int64'})

    player_df = player_df[feature_names]

    # May need to add imputer for data transformation!
    id_train = (player_df.season < testing_season)  # & (pbp_df.period <= 4) #| (win_df.week < 22))
    id_test  = (player_df.season == testing_season)  # & (pbp_df.period <= 4) # & (~win_df['winProbability.before'].isnull())

    features_train_player = player_df.loc[id_train]
    features_test_player  = player_df.loc[id_test]

    print(features_train_player.shape, features_test_player.shape)

    # for classification -> label needs to be put into integer categories
    label = player_df.points
    label = label.astype('int32')
    label.loc[label > 45] = 45

    label_train_player = label[id_train]
    label_test_player = label[id_test]

    return features_train_player, features_test_player, label_train_player, label_test_player

def query_team_stats():
    team_stats_query_string = open("src/queries/team_stats.sql", 'r').read()
    stats_df = query_athena(team_stats_query_string)
    print('team_stats query is done!')

    # self join to create opponent parameters!
    stats_df_opp = stats_df.drop(columns=['opp_team_id','event_type_id','at_home'])
    id = stats_df_opp.columns.isin(['season','game_code','date','team_id'])
    stats_df_opp.rename(columns={name:name+"_conceded" for name in stats_df_opp.columns[~id]}, inplace=True)
    stats_df_opp.rename(columns={'team_id':'opp_team_id'}, inplace=True)

    stats_df = pd.merge(stats_df, stats_df_opp, how='left', on=['season','game_code','date','opp_team_id'])

    # groupby previous - seasonal average results
    season_df = stats_df.drop(columns=['game_code', 'event_type_id', 'date', 'at_home', 'opp_team_id'])
    season_df['season'] = season_df['season'] + 1
    season_df = season_df.loc[season_df.season <= 2018]
    gd = season_df.groupby(['season', 'team_id'])
    season_df = gd.mean().reset_index()

    # Use previous seasonal average to create initial values for current season
    tmp = pd.DataFrame({'date': ['00', '01', '02', '03', '04']})
    tmp['key'] = 0
    season_df['key'] = 0
    initial_df = season_df.merge(tmp, how='left', on='key').drop(columns=['key'])
    initial_df.head(11)

    # combine initial values with current season game values
    stats_df = stats_df.loc[stats_df.season > 2004]
    stats_df = pd.concat([initial_df, stats_df], axis=0)
    stats_df.sort_values(by=['season', 'team_id', 'date'], axis=0, inplace=True)

    # calculate moving avg of last 5 games
    id = stats_df.columns.isin(['season', 'team_id', 'opp_team_id', 'game_code', 'event_type_id', 'at_home', 'date'])
    stats_df.rename(columns={name: name + '_l5' for name in stats_df.columns[~id]}, inplace=True)

    gd = stats_df.groupby(by=['season', 'team_id'])
    stats_df.loc[:, ~id] = gd.apply(lambda df: df.loc[:, ~id].rolling(5).mean().shift(1))

    stats_df = stats_df.loc[stats_df.date > '04', :]

    # stats_df add game index for each team
    gd = stats_df.groupby(['season', 'team_id'])
    stats_df['game_num'] = gd.cumcount() + 1

    return stats_df

def load_elo_ratings():
    # Note:
    #  > in elo_data, team_1 is home team
    #  > data issue in 2007, MIA vs ATL on 2007-12-19 was mistakenly placed on 2008-03-07, thus mess up whole season elo
    elo_data = pd.read_csv('runtime/datasets/nba_elo.csv')
    elo_data.season = elo_data.season - 1
    elo_data = elo_data[(elo_data.season >= 2005) & (elo_data.season <= 2018) & (elo_data.season != 2007)]

    elo_data = elo_data[['date', 'season', 'team1', 'team2', 'elo1_pre', 'elo2_pre']]

    # add team_id
    team_dict_df = pd.DataFrame(pd.Series(team_dict)).reset_index()
    team_dict_df.columns = ['team', 'team_id']

    elo_data = pd.merge(elo_data, team_dict_df, left_on='team1', right_on='team', how='left')
    elo_data.drop('team', axis=1, inplace=True)
    elo_data.rename(columns={'team_id': 'team1_id'}, inplace=True)

    elo_data = pd.merge(elo_data, team_dict_df, left_on='team2', right_on='team', how='left')
    elo_data.drop('team', axis=1, inplace=True)
    elo_data.rename(columns={'team_id': 'team2_id'}, inplace=True)

    # add game_num for each team
    # concat game team pair into one team dataframe
    df_1 = elo_data.loc[:, ['date', 'season', 'team1_id']]
    df_2 = elo_data.loc[:, ['date', 'season', 'team2_id']]

    df_1.rename(columns={'team1_id': 'team_id'}, inplace=True)
    df_2.rename(columns={'team2_id': 'team_id'}, inplace=True)

    df = pd.concat([df_1, df_2], axis=0).sort_values(by=['season', 'team_id', 'date'])

    gd = df.groupby(['season', 'team_id'])
    df['game_num'] = gd.cumcount() + 1

    # merge game_num back into elo_data
    elo_data = pd.merge(elo_data, df, how='left', left_on=['date', 'season', 'team1_id'],
                        right_on=['date', 'season', 'team_id'])
    elo_data.rename(columns={'game_num': 'team1_game_num'}, inplace=True)
    elo_data.drop(columns=['team_id'], inplace=True)

    elo_data = pd.merge(elo_data, df, how='left', left_on=['date', 'season', 'team2_id'],
                        right_on=['date', 'season', 'team_id'])
    elo_data.rename(columns={'game_num': 'team2_game_num'}, inplace=True)
    elo_data.drop(columns=['team_id'], inplace=True)

    return elo_data

def create_game_info(stats_df, elo_data):
    game_df = pd.merge(elo_data, stats_df.drop(columns=['date', 'opp_team_id']), how='left',
                       left_on=['season', 'team1_id', 'team1_game_num'],
                       right_on=['season', 'team_id', 'game_num'])
    game_df.drop(columns=['game_num', 'team_id'], inplace=True)

    col_names = ['points_l5', 'fg_attempted_l5', 'fg_made_l5',
                 'ft_attempted_l5', 'ft_made_l5', 'offensive_rebounds_l5',
                 'defensive_rebounds_l5', 'turnovers_l5', 'assists_l5',
                 'points_conceded_l5',
                 'fg_attempted_conceded_l5', 'fg_made_conceded_l5',
                 'ft_attempted_conceded_l5', 'ft_made_conceded_l5',
                 'offensive_rebounds_conceded_l5', 'defensive_rebounds_conceded_l5',
                 'turnovers_conceded_l5', 'assists_conceded_l5'
                 ]
    game_df.rename(columns={name: ('t1_' + name) for name in col_names}, inplace=True)

    game_df = pd.merge(game_df, stats_df.drop(columns=['date', 'game_code', 'event_type_id', 'at_home', 'opp_team_id']),
                       how='left',
                       left_on=['season', 'team2_id', 'team2_game_num'],
                       right_on=['season', 'team_id', 'game_num'])

    game_df.drop(columns=['game_num', 'team_id'], inplace=True)

    game_df.rename(columns={name: ('t2_' + name) for name in col_names}, inplace=True)

    return game_df

def query_player_stats():
    player_stats_query_string = open("src/queries/player_data.sql", 'r').read()
    player_df = query_athena(player_stats_query_string)
    print('team_stats query is done!')

    # groupby previous - seasonal player average results
    player_season_df = player_df.drop(
        columns=['game_code', 'game_time', 'position_id', 'team_id', 'at_home', 'draft_year',
                 'game_started'])
    gd = player_season_df.groupby(['season', 'player_id'])
    player_season_df = gd.mean().reset_index()

    player_season_df['season'] = player_season_df['season'] + 1
    player_season_df = player_season_df.loc[player_season_df.season <= 2018]

    # Use previous seasonal average to create initial values for current season
    tmp = pd.DataFrame({'game_time': [*range(10)], 'game_started': [False] * 10})

    # This is called cross-join!
    tmp['key'] = 0
    player_season_df['key'] = 0
    initial_df = player_season_df.merge(tmp, how='left', on='key').drop(columns=['key'])

    # combine initial values with current season game values
    player_df = player_df.loc[player_df.season > 2004]
    player_df = pd.concat([initial_df, player_df], axis=0)
    player_df.sort_values(by=['season', 'player_id', 'game_time'], axis=0, inplace=True)
    player_df.reset_index(inplace=True, drop=True)

    # calculate moving avg of last 10 games (only when the player actually played)
    # special case I: a player in last season, but not this season, will be filtered out
    # .             II: a rookie player -> no previous season data, so we will give up the 1st 10 games in modeling
    #                  , but in production system, we may want human inputs as the initial guess

    id = player_df.columns.isin(['minutes',
                                 'points', 'fg_attempt', 'fg_made', 'ft_attempt', 'ft_made',
                                 'point_3_attempt', 'point_3_made', 'offensive_rebounds',
                                 'defensive_rebounds', 'assists', 'blocks', 'turnovers'])

    gd = player_df.groupby(by=['season', 'player_id'])
    tmp = gd.apply(lambda df: df.loc[:, id].rolling(10).mean().shift(1))

    tmp.rename(columns={name: name + '_l10' for name in tmp.columns}, inplace=True)

    player_df = pd.concat([player_df, tmp], axis=1)

    player_df = player_df.loc[~player_df.minutes_l10.isnull()]

    # calculate rates per min
    rates_name = ['points_l10', 'fg_attempt_l10', 'fg_made_l10', 'ft_attempt_l10',
                  'ft_made_l10', 'point_3_attempt_l10', 'point_3_made_l10',
                  'offensive_rebounds_l10', 'defensive_rebounds_l10', 'assists_l10',
                  'blocks_l10', 'turnovers_l10']
    player_df.loc[:, rates_name] = player_df.loc[:, rates_name].div(player_df.minutes_l10, axis='index')

    return player_df

def combine_team_and_player_info(game_df, player_df):
    # merge team stats/elo into game_df
    tmp1 = game_df.loc[:, ['season', 'game_code', 'team1_id', 'team2_id', 'elo1_pre',
                           't1_offensive_rebounds_l5', 't1_defensive_rebounds_l5',
                           't1_offensive_rebounds_conceded_l5', 't1_defensive_rebounds_conceded_l5']]

    tmp2 = game_df.loc[:, ['season', 'game_code', 'team2_id', 'team1_id', 'elo2_pre',
                           't2_offensive_rebounds_l5', 't2_defensive_rebounds_l5',
                           't2_offensive_rebounds_conceded_l5', 't2_defensive_rebounds_conceded_l5']]

    tmp1.columns = tmp2.columns = ['season', 'game_code', 'team_id', 'opp_id', 'team_elo',
                                   'team_o_rebounds_l5', 'team_d_rebounds_l5',
                                   'team_o_rebounds_conceded_l5', 'team_d_rebounds_conceded_l5']

    tmp3 = pd.concat([tmp1, tmp2], axis=0)

    # adding features related to player's team
    player_df = player_df[player_df.season!=2007]
    player_df = pd.merge(player_df, tmp3, how='left',
                           left_on=['season','game_code','team_id'],
                           right_on=['season','game_code','team_id'])

    tmp4 = tmp3.drop(columns=['opp_id'])
    tmp4.columns = ['season', 'game_code', 'opp_id', 'opp_elo',
                       'opp_o_rebounds_l5', 'opp_d_rebounds_l5',
                       'opp_o_rebounds_conceded_l5', 'opp_d_rebounds_conceded_l5']

    # adding features related to player's opponent team
    player_df = pd.merge(player_df, tmp4, how='left',
                           left_on=['season','game_code','opp_id'],
                           right_on=['season','game_code','opp_id'])

    return player_df

def generate_dataset_local():

    print('start querying ... ')

    # (1) parse team stats
    team_stats = query_team_stats()

    # (2) elo ratings
    elo_df = load_elo_ratings()

    # (3) combine elo-rating with team-stats
    game_df = create_game_info(team_stats, elo_df)

    # (4) query player data
    player_df = query_player_stats()

    # (5) merge team_stats/elo into player_df
    player_df = combine_team_and_player_info(game_df, player_df)

    # (6) generate features and labels
    features_train, features_test, label_train, label_test = generate_features(player_df, TESTING_SEASON)

    save_dataset(features_train, "features_player_train")

    save_dataset(features_test,  "features_player_test")

    save_dataset(label_train, "label_player_train")

    save_dataset(label_test,  "label_player_test")

    return

if __name__ == '__main__':
    print('hello')
    generate_dataset_local()
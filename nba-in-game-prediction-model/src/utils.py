from pyathena import connect

import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

import mlflow


team_dict = {'ATL': 1, 'BOS': 2, 'BRK': 17, 'CHI': 4, 'CHO': 5312, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8,
             'GSW': 9, 'HOU': 10, 'IND': 11, 'LAC': 12, 'LAL': 13, 'MEM': 29, 'MIA': 14, 'MIL': 15, 'MIN': 16,
             'NJN': 17, 'NOK': 3, 'NOP': 3, 'NYK': 18, 'OKC': 25, 'ORL': 19, 'PHI': 20, 'PHO': 21, 'POR': 22,
             'SAC': 23, 'SAS': 24, 'SEA': 25, 'TOR': 28, 'UTA': 26, 'WAS': 27}

TESTING_SEASON=2018

EXPERIMENT_NAME = 'nba-player-props-season-simulation'

MODEL = 'nba-player-props-season-simulation'
MODEL_VERSION = '0.0.01'

hyperparams = {
    'solver':'lbfgs',
    'fit_intercept':True,
    'C':0.1,
    'max_iter':100000,
    'tol':1e-5,
    'random_state':42
}

num_fields_player = [
    # player stats
    'minutes_l10',
    'points_l10',
    'fg_attempt_l10', 'fg_made_l10',
    'ft_attempt_l10', 'ft_made_l10',
    'point_3_attempt_l10', 'point_3_made_l10',
    'offensive_rebounds_l10', 'defensive_rebounds_l10',
    'assists_l10',
    'blocks_l10',
    'turnovers_l10',

    # team stats
    'team_o_rebounds_l5', 'team_d_rebounds_l5',
    'team_o_rebounds_conceded_l5', 'team_d_rebounds_conceded_l5',
    'opp_o_rebounds_l5', 'opp_d_rebounds_l5',
    'opp_o_rebounds_conceded_l5', 'opp_d_rebounds_conceded_l5',

    # team elo
    'team_elo',
    'opp_elo',
]

cat_fields_player = ['at_home', 'game_started']

feature_names = num_fields_player + cat_fields_player + ['season','team_id','player_id','game_code','points']

def query_athena(query_string):
    conn = connect(
        s3_staging_dir='s3://aws-athena-query-results-323906537337-us-east-1/',
        region_name='us-east-1'
    )
    return pd.read_sql(query_string, conn)

def local_data_path():
    return os.path.abspath('runtime/datasets')

def save_dataset(data, name):
    if not os.path.exists(local_data_path()):
        os.makedirs(local_data_path())
    data.to_csv(os.path.join(local_data_path(), name + '.csv'), index=False)
    #np.savetxt(os.path.join(local_data_path(), name + '.csv'), data, delimiter=',')

def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')

def load_dataset_prediction(data_name):
    return pd.read_csv('../runtime/datasets' + '/'+ data_name + '.csv')

def model_train(model_hyperparams, model_version, features, labels, transform_pipeline):

    if model_version in ['0.0.01']:
        model = LogisticRegression(**model_hyperparams)
    elif model_version in ['0.0.02']:
        model = LinearSVC(**model_hyperparams)
    elif model_version in ['0.0.03']:
        model = RandomForestClassifier(**model_hyperparams)
    elif model_version in ['0.0.04']:
        model = RandomForestRegressor(**model_hyperparams)
    # elif model_version in ['0.0.10']:
    #     features = transform_pipeline.fit_transform(features)
    #     #model = KerasClassifier(build_fn=create_DL_classifier_model, epochs=70, verbose=0)
    #     #model = KerasClassifier(build_fn=create_DL_classifier_model, **model_hyperparams)
    #     model = create_DL_classifier_model(features.shape[1])
    else:
        print("Error: wrong MODEL_VERSION:", model_version)
        raise Exception("Wrong model_version for model_train function call")

    pipe = Pipeline([('columnTransfer', transform_pipeline), ('model', model)])
    pipe.fit(features, labels)

    return pipe

def evaluate_classification(model, model_version, features_train, label_train, features_test, label_test):
    if mlflow.active_run() is None:
        raise Exception('Cannot call evaluate() method without first setting an active mlflow run.')

    if model_version in ['0.0.10']:
        features_train = model[0].transform(features_train)
        features_test  = model[0].transform(features_test)
        train_probs = model[1].predict(features_train)
        test_probs  = model[1].predict(features_test)
    else:
        train_probs = model.predict_proba(features_train)
        test_probs = model.predict_proba(features_test)

    #print('mean: ', train_probs.mean(), test_probs.mean())

    train_roc_auc = roc_auc_score(label_train, train_probs, multi_class='ovr')
    test_roc_auc  = roc_auc_score(label_test, test_probs, multi_class='ovr')
    mlflow.log_metric('train_roc_auc', train_roc_auc)
    mlflow.log_metric('test_roc_auc',  test_roc_auc)
    print('roc: ', train_roc_auc, test_roc_auc)

    train_mae = mean_absolute_error(label_train, np.dot(train_probs, np.array(range(46))))
    test_mae = mean_absolute_error(label_test, np.dot(test_probs, np.array(range(46))))
    mlflow.log_metric('train_mae', train_mae)
    mlflow.log_metric('test_mae', test_mae)
    print('mae: ', train_mae, test_mae)

    train_mse = mean_squared_error(label_train, np.dot(train_probs, np.array(range(46))))
    test_mse  = mean_squared_error(label_test, np.dot(test_probs, np.array(range(46))))
    train_rmse = np.sqrt(train_mse)
    test_rmse  = np.sqrt(test_mse)
    mlflow.log_metric('train_rmse', train_rmse )
    mlflow.log_metric('test_rmse', test_rmse )
    print('rmse: ', train_rmse, test_rmse)

    #predeval.plot_calibration_curve(label_test, test_probs, stat_name='in-game-win-prob')

    return

import pandas as pd
import numpy as np
import os
import mlflow


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from pyathena import connect
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

import stats.predeval as predeval

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

TRAINING_END_SEASON = 2018

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


def create_DL_classifier_model(input_n):
    model_DL = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[input_n, ]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(250, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(150, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(80, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(30, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # multi-nomial classification
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['AUC'])

    # binary classification
    optimizer = keras.optimizers.SGD(learning_rate=0.01, decay=1e-4)
    model_DL.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[['accuracy', 'AUC']])

    # model_DL.summary()

    return model_DL


def model_train(model_hyperparams, model_version, features, labels, transform_pipeline):
    if model_version in ['0.0.01']:
        model = LogisticRegression(**model_hyperparams)
    elif model_version in ['0.0.02']:
        model = LinearSVC(**model_hyperparams)
    elif model_version in ['0.0.03']:
        model = RandomForestClassifier(**model_hyperparams)
    elif model_version in ['0.0.04']:
        model = RandomForestRegressor(**model_hyperparams)
    elif model_version in ['0.0.10']:
        features = transform_pipeline.fit_transform(features)
        #model = KerasClassifier(build_fn=create_DL_classifier_model, epochs=70, verbose=0)
        #model = KerasClassifier(build_fn=create_DL_classifier_model, **model_hyperparams)
        model = create_DL_classifier_model(features.shape[1])
    else:
        print("Error: wrong MODEL_VERSION:", model_version)
        raise Exception("Wrong model_version for model_train function call")

    if model_version in ['0.0.10']:
        # this is only a tmp solution!
        model.fit(features, labels, **model_hyperparams)
        pipe = (transform_pipeline, model)
    else:
        pipe = Pipeline([('columnTransfer', transform_pipeline), ('model', model)])
        pipe.fit(features, labels)

    return pipe


# feature_train, label_train, feature_test, label_test: string represent file name
def evaluate_classification(model, model_version, features_train, label_train, features_test, label_test):
    if mlflow.active_run() is None:
        raise Exception('Cannot call evaluate() method without first setting an active mlflow run.')

    if model_version in ['0.0.10']:
        features_train = model[0].transform(features_train)
        features_test  = model[0].transform(features_test)
        train_probs = model[1].predict(features_train)
        test_probs  = model[1].predict(features_test)
    else:
        train_probs = pd.Series(model.predict_proba(features_train)[:, 1])
        test_probs = pd.Series(model.predict_proba(features_test)[:, 1])

    print('mean: ', train_probs.mean(), test_probs.mean())

    train_roc_auc = roc_auc_score(label_train, train_probs)
    test_roc_auc  = roc_auc_score(label_test, test_probs)
    mlflow.log_metric('train_roc_auc', train_roc_auc)
    mlflow.log_metric('test_roc_auc',  test_roc_auc)
    print('roc: ', train_roc_auc, test_roc_auc)

    #predeval.plot_calibration_curve(label_test, test_probs, stat_name='in-game-win-prob')

    return


def evaluate_multi_classification(model, model_version, features_train, label_train, features_test, label_test):
    if mlflow.active_run() is None:
        raise Exception('Cannot call evaluate() method without first setting an active mlflow run.')

    train_probs = pd.DataFrame(model.predict_proba(features_train))
    test_probs  = pd.DataFrame(model.predict_proba(features_test))

    print('mean: \n', train_probs.mean(), '\n', test_probs.mean())

    train_roc_auc = roc_auc_score(label_train, train_probs, multi_class='ovr')
    test_roc_auc  = roc_auc_score(label_test, test_probs, multi_class='ovr')
    mlflow.log_metric('train_roc_auc', train_roc_auc)
    mlflow.log_metric('test_roc_auc',  test_roc_auc)
    print('roc: ', train_roc_auc, test_roc_auc)

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


def calculate_drive_outcome(pbp_df):
    eScoreDiff = pbp_df.groupby(['game_code', 'drive_id', 'offense_team'])[
        ['game_code', 'drive_id', 'offense_team', 'score_diff_after']].tail(1). \
        reset_index(drop=True)
    sScoreDiff = pbp_df.groupby(['game_code', 'drive_id', 'offense_team'])[
        ['game_code', 'drive_id', 'offense_team', 'score_diff']].head(1).reset_index(drop=True)
    eScoreDiff['drive_score_diff_change'] = (eScoreDiff.score_diff_after - sScoreDiff.score_diff)
    eScoreDiff['drive_outcome'] = 0  # no-score
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == 1, 'drive_outcome'] = 1  # FG
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == 2, 'drive_outcome'] = 2  # FG
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == 3, 'drive_outcome'] = 3  # FG
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == 6, 'drive_outcome'] = 4  # TD w.o. extra point
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == 7, 'drive_outcome'] = 5  # TD w extra point
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == 8, 'drive_outcome'] = 6  # TD w extra point
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == -2, 'drive_outcome'] = 7  # defense points from offense turn-over
    eScoreDiff.loc[eScoreDiff.drive_score_diff_change == -6, 'drive_outcome'] = 8  # defense points from offense turn-over

    pbp_df = pd.merge(pbp_df,
                      eScoreDiff[['game_code', 'drive_id', 'offense_team', 'drive_score_diff_change', 'drive_outcome']],
                      on=['game_code', 'drive_id', 'offense_team'])
    sScoreDiff.rename(columns={'score_diff': 'drive_start_score_diff'}, inplace=True)
    pbp_df = pd.merge(pbp_df, sScoreDiff[['game_code', 'drive_id', 'offense_team', 'drive_start_score_diff']],
                      on=['game_code', 'drive_id', 'offense_team'])
    return pbp_df


def calculate_remaining_timeout(pbp_df):
    pbp_df[['home_timeout', 'away_timeout', 'remaining_home_timeouts', 'remaining_home_timeouts']] = 0

    id = ((pbp_df.offense_timeout == 1) & (pbp_df.offense_team == pbp_df.home_team)) | \
         ((pbp_df.defense_timeout == 1) & (pbp_df.defense_team == pbp_df.home_team))
    pbp_df.loc[id, 'home_timeout'] = 1
    id = ((pbp_df.offense_timeout == 1) & (pbp_df.offense_team == pbp_df.away_team)) | \
         ((pbp_df.defense_timeout == 1) & (pbp_df.defense_team == pbp_df.away_team))
    pbp_df.loc[id, 'away_timeout'] = 1

    gd = pbp_df.groupby(['game_code', 'half_game'])
    pbp_df[['remaining_home_timeouts', 'remaining_away_timeouts']] = 3 - gd[['home_timeout', 'away_timeout']].cumsum()

    pbp_df[['remaining_offense_timeouts', 'remaining_defense_timeouts']] = \
        pbp_df[['remaining_home_timeouts', 'remaining_away_timeouts']].copy()
    id = (pbp_df.offense_team == pbp_df.away_team)
    pbp_df.loc[id, 'remaining_offense_timeouts'] = pbp_df.loc[id, 'remaining_away_timeouts'].values.astype('float64')
    pbp_df.loc[id, 'remaining_defense_timeouts'] = pbp_df.loc[id, 'remaining_home_timeouts'].values.astype('float64')


def calculate_remaining_timeout_cfb(pbp_df):
    pbp_df[['home_timeout', 'away_timeout', 'remaining_home_timeouts', 'remaining_home_timeouts']] = 0
    id = ((pbp_df.offense_timeout == 1) & (pbp_df.offense_team == pbp_df.home_team)) | \
         ((pbp_df.defense_timeout == 1) & (pbp_df.defense_team == pbp_df.home_team))
    pbp_df.loc[id, 'home_timeout'] = 1
    id = ((pbp_df.offense_timeout == 1) & (pbp_df.offense_team == pbp_df.away_team)) | \
         ((pbp_df.defense_timeout == 1) & (pbp_df.defense_team == pbp_df.away_team))
    pbp_df.loc[id, 'away_timeout'] = 1
    gd = pbp_df.groupby(['game_code', 'half_game'])
    pbp_df[['remaining_home_timeouts', 'remaining_away_timeouts']] = gd[['home_timeout', 'away_timeout']].cumsum()
    id = pbp_df.period <= 4
    pbp_df.loc[id, ['remaining_home_timeouts', 'remaining_away_timeouts']] = \
        3 - pbp_df[['remaining_home_timeouts', 'remaining_away_timeouts']]
    pbp_df.loc[~id, ['remaining_home_timeouts', 'remaining_away_timeouts']] = \
        1 - pbp_df[['remaining_home_timeouts', 'remaining_away_timeouts']]
    pbp_df[['remaining_offense_timeouts', 'remaining_defense_timeouts']] = \
        pbp_df[['remaining_home_timeouts', 'remaining_away_timeouts']].copy()
    id = (pbp_df.offense_team == pbp_df.away_team)
    pbp_df.loc[id, 'remaining_offense_timeouts'] = pbp_df.loc[id, 'remaining_away_timeouts'].values.astype('float64')
    pbp_df.loc[id, 'remaining_defense_timeouts'] = pbp_df.loc[id, 'remaining_home_timeouts'].values.astype('float64')
    pbp_df.loc[pbp_df.remaining_offense_timeouts < 0, 'remaining_offense_timeouts'] = 0
    pbp_df.loc[pbp_df.remaining_defense_timeouts < 0, 'remaining_defense_timeouts'] = 0


def calculate_score_diff(pbp_df):
    pbp_df['score_diff'] = pbp_df.home_score - pbp_df.away_score
    pbp_df.loc[pbp_df.offense_team == pbp_df.away_team, 'score_diff'] = pbp_df.away_score - pbp_df.home_score
    pbp_df.score_diff = pbp_df.score_diff.astype('float64')  # important for num features
    pbp_df['score_diff_after'] = pbp_df.home_score_after - pbp_df.away_score_after
    pbp_df.loc[pbp_df.offense_team == pbp_df.away_team, 'score_diff_after'] = pbp_df.away_score_after - pbp_df.home_score_after
    pbp_df.score_diff_after = pbp_df.score_diff_after.astype('float64')  # important for num features

    pbp_df['adj_score_diff'] = pbp_df.score_diff / np.power(pbp_df.remaining_game_time + 1, 0.5)


def calculate_adj_expected_score_diff(drive_outcome_p, features):
    #global expectedDriveScore
    expected_drive_score = np.dot(drive_outcome_p, np.array([0, 1, 2, 3, 6, 7, 8, -2, -6]))
    features['expected_extra_score'] = features.drive_start_score_diff - features.score_diff + \
                                           expected_drive_score
    features['adj_expected_score_diff'] = (features.expected_extra_score + features.score_diff) / \
                                             np.power(features.remaining_game_time + 1, 0.5)
import pandas as pd
import os

EXPERIMENT_NAME = 'nfl-usage-team-offensive-plays'
MODEL_VERSION = '0.0.05'

# hyperparams = {
#     'n_estimators': 200,
#     'max_depth': 3,
#     'random_state': 42
# }

hyperparams = {
    'max_iter': 10000,
    'tol': 1e-6,
    'random_state': 42
}

featureNames = ['ytd_offensivePlaysAdj', 'ytd_paceAdj', 'ytd_paceConcededAdj']

num_fields = ['ytd_offensivePlaysAdj', 'ytd_paceAdj', 'ytd_paceConcededAdj']

alpha = 0.9995
alpha_team = 0.99
playsThreshold = 10
predictProb = False


def local_data_path():
    return os.path.abspath('../../runtime/datasets')


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')
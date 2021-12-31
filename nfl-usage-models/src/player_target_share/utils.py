import pandas as pd
import os

EXPERIMENT_NAME = 'nfl-usage-player-target-share'
MODEL_VERSION = '0.0.07'

hyperparams = {
    # 'max_iter': 10000,
    # 'tol': 1e-6,
    # 'random_state': 42
}

featureNames = ['ytd_targetShareAdj', 'ytd_targetShareByPositionRankAdj', 'ytd_rank', 'positionid']

num_fields = ['ytd_targetShareAdj', 'ytd_rank']
cat_fields = ['positionid']

predictProb = False

def local_data_path():
    return os.path.abspath('../../runtime/datasets')


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')

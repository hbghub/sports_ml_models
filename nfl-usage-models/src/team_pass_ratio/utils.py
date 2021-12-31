import pandas as pd
import os

EXPERIMENT_NAME = 'nfl-usage-team-pass-ratio'
MODEL_VERSION = '0.0.07'


hyperparams = {
    # 'max_iter': 10000,
    # 'tol': 1e-6,
    # 'random_state': 42
}

featureNames = ['ytd_passRatioAdj',
                  #'ytd_scrambles',
                  'ytd_scrambleRatio',
                  #'odds',
                  'favoritePoints',
                  'ytd_passingYardsPerAttempt',
                  'ytd_rushingYardsPerAttempt',
                  'ytd_o_passingYardsPerAttempt',
                  'ytd_o_rushingYardsPerAttempt',
                  'ytd_passingYardsPerAttempt_conceded',
                  'ytd_rushingYardsPerAttempt_conceded',
                  'ytd_o_passingYardsPerAttempt_conceded',
                  'ytd_o_rushingYardsPerAttempt_conceded',
                  ]

num_fields = ['ytd_passRatioAdj',
                  #'ytd_scrambles',
                  'ytd_scrambleRatio',
                  #'odds',
                  'favoritePoints',
                  'ytd_passingYardsPerAttempt',
                  'ytd_rushingYardsPerAttempt',
                  'ytd_o_passingYardsPerAttempt',
                  'ytd_o_rushingYardsPerAttempt',
                  'ytd_passingYardsPerAttempt_conceded',
                  'ytd_rushingYardsPerAttempt_conceded',
                  'ytd_o_passingYardsPerAttempt_conceded',
                  'ytd_o_rushingYardsPerAttempt_conceded',
                  ]

predictProb = False


def local_data_path():
    return os.path.abspath('../../runtime/datasets')


def load_dataset(data_name):
    return pd.read_csv(local_data_path() + '/'+ data_name + '.csv')

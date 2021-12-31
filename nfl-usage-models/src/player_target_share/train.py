#!/usr/bin/env python

import argparse
import sys

from src.utils import load_dataset, model_train, evaluate_regression
from src.player_target_share.utils import MODEL_VERSION, EXPERIMENT_NAME, hyperparams, featureNames, num_fields, cat_fields

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import mlflow
import mlflow.sklearn

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow_tags = {
    'product': "Predictions",  # str or list[str]: fill in related products(s)
    'data': "Event",  # str or list[str]: fill in data(m) the model uses
    'sport': "Football",  # str or list[str]: fill in related sport(s)
    'short_description': "nfl rates model based on xinfo pbp data",  # str: fill in single sentence description of your model
    'documentation_url': "https://bitbucket.org/statsinc/nfl-rates-pbp/src/master/README.md"  # str or list[str]: fill in link(s) to documentation of your model
}


def get_command_line_arguments(args, return_dict: bool = True):
    parser = argparse.ArgumentParser(
        description="Command line arguments for training script."
    )
    parser.add_argument(
        "-bs", "--batch-size",
        default=256,
        type=int,
        help="number of examples per training batch."
    )
    parser.add_argument(
        "-ne", "--num-epochs",
        default=10,
        type=int,
        help="number of complete passes through the training dataset."
    )
    parser.add_argument(
        "-hu", "--hidden-units",
        nargs="+",
        type=int,
        help="number and size of network hidden units, e.g. '--hu 1 2 3 4'."
    )
    parser.add_argument(
        "-md", "--model-dir",
        type=str,
        help="directory to export/save model to."
    )
    args = parser.parse_args(args=args)

    if return_dict:
        return {k: v for k, v in vars(args).items() if v is not None}

    return args


if __name__ == "__main__":
    params = get_command_line_arguments(
        args=sys.argv[1:],
        return_dict=True
    )
    with mlflow.start_run() as run:
        print(f'MODEL_VERSION: {MODEL_VERSION}')
        print(f'MLflow experiment ID: {mlflow.active_run().info.experiment_id}')
        print(f'MLflow run ID: {mlflow.active_run().info.run_id}')

        features_train = load_dataset('features_target_share_train')
        label_train = load_dataset('label_target_share_train').values.ravel()
        #pbp_df = load_dataset('pbp_df')

        features_train = features_train[featureNames]

        # feature adjustment for injury consideration
        id = features_train.ytd_targetShareByPositionRankAdj > 0
        features_train.loc[id,'ytd_targetShareAdj'] = features_train.ytd_targetShareByPositionRankAdj[id]
        features_train.drop(columns=['ytd_targetShareByPositionRankAdj'], inplace=True)

        transform_pipeline = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_fields),
            ('cat', OneHotEncoder(categories='auto'), cat_fields)
        ])

        model = model_train(hyperparams, MODEL_VERSION, features_train, label_train, transform_pipeline)

        features_test = load_dataset('features_target_share_test')[featureNames]
        label_test = load_dataset('label_target_share_test').values.ravel()

        id = features_test.ytd_targetShareByPositionRankAdj > 0
        features_test.loc[id,'ytd_targetShareAdj'] = features_test.ytd_targetShareByPositionRankAdj[id]
        features_test.drop(columns=['ytd_targetShareByPositionRankAdj'], inplace=True)

        evaluate_regression(model, features_train, label_train, features_test, label_test)

        mlflow.set_tags(mlflow_tags)
        mlflow.log_params(hyperparams)
        mlflow.log_param('model_version', MODEL_VERSION)

        #mlflow.log_artifact('runtime/results/')
        mlflow.sklearn.log_model(model, EXPERIMENT_NAME, registered_model_name=EXPERIMENT_NAME)
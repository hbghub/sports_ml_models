#!/usr/bin/env python

import argparse
import sys
import pandas as pd
import numpy  as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import load_dataset, model_train, evaluate_multi_classification
from src.overtime_baseline_model.utils import MODEL_VERSION, MODEL_TEAM_DRIVE_VERSION, MODEL, MODEL_TEAM_DRIVE,\
                                                EXPERIMENT_NAME, hyperparams, hyperparams_team_drive, featureNames, \
                                                num_fields, cat_fields, cat_fields_drive

import mlflow
import mlflow.sklearn

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow_tags = {
    'product': "Predictions",  # str or list[str]: fill in related products(s)
    'data': "Event",  # str or list[str]: fill in data(m) the model uses
    'sport': "Football",  # str or list[str]: fill in related sport(s)
    'short_description': "nfl in-game win probability model",  # str: fill in single sentence description of your model
    'documentation_url': "https://bitbucket.org/statsinc/nfl-win-probability/src/master/README.md"  # str or list[str]: fill in link(s) to documentation of your model
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
    #if True:
        features_train = load_dataset('features_win_prob_ot_baseline_train')[featureNames]
        label_train = load_dataset('label_win_prob_ot_baseline_train').values.ravel()

        transform_pipeline = ColumnTransformer(transformers=[
            ('num', 'passthrough', num_fields),
            ('cat', OneHotEncoder(categories='auto'), cat_fields)
        ])

        transform_pipeline_drive = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(categories='auto'), cat_fields_drive)
        ])

        model = model_train(hyperparams, MODEL_VERSION, features_train, label_train, transform_pipeline)

        features_test = load_dataset('features_win_prob_ot_baseline_test')[featureNames]
        label_test = load_dataset('label_win_prob_ot_baseline_test').values.ravel()

        print(features_train.shape, features_test.shape)

        evaluate_multi_classification(model, MODEL_VERSION, features_train, label_train, features_test, label_test)

        # train final model
        features = pd.concat((features_train, features_test), axis=0)
        label    = np.concatenate((label_train, label_test), axis=0)

        model = model_train(hyperparams, MODEL_VERSION, features, label, transform_pipeline)

        features['drive_outcome'] = label
        features_drive = features.groupby(['game_code','drive_id','offense_team']).head(1)
        label_drive = features_drive.drive_outcome
        features_drive.drop('drive_outcome', axis=1, inplace=True)
        model_team_drive = model_train(hyperparams_team_drive, MODEL_TEAM_DRIVE_VERSION,
                                  features_drive, label_drive, transform_pipeline_drive)

        mlflow.set_tags(mlflow_tags)

        mlflow.log_param('drive_model_version', MODEL_VERSION)
        mlflow.log_param('drive_model', hyperparams)
        mlflow.log_param('team_drive_model_version', MODEL_TEAM_DRIVE_VERSION)
        mlflow.log_param('team_drive_model', hyperparams_team_drive)

        #mlflow.log_artifact('runtime/results/')
        mlflow.sklearn.log_model(model, MODEL, registered_model_name=MODEL)
        mlflow.sklearn.log_model(model_team_drive, MODEL_TEAM_DRIVE, registered_model_name=MODEL_TEAM_DRIVE)



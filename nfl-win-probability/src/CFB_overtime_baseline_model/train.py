#!/usr/bin/env python

import argparse
import sys

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import load_dataset, model_train, evaluate_multi_classification
from src.CFB_overtime_baseline_model.utils import MODEL, MODEL_VERSION, EXPERIMENT_NAME,\
    hyperparams, featureNames, num_fields, cat_fields

import mlflow
import mlflow.sklearn

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow_tags = {
    'product': "Predictions",  # str or list[str]: fill in related products(s)
    'data': "Event",  # str or list[str]: fill in data(m) the model uses
    'sport': "Football",  # str or list[str]: fill in related sport(s)
    'short_description': "CFB football in-game win probability model",  # str: fill in single sentence description of your model
    'documentation_url': "https://bitbucket.org/statsinc/us-football-win-probability/src/master/README.md"  # str or list[str]: fill in link(s) to documentation of your model
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
        # temporarily share the data with reg time game
        features_train = load_dataset('features_win_prob_CFB_ot_train')[featureNames]
        label_drive_train = load_dataset('label_win_prob_CFB_ot_train').values.ravel()

        # parse label for drive outcome and fit the model for the new features
        #label_drive_train = features_train.drive_outcome

        transform_pipeline_drive = ColumnTransformer(transformers=[
            ('num', 'passthrough', num_fields),
            ('cat', OneHotEncoder(categories='auto'), cat_fields)
        ])

        model_drive = model_train(hyperparams, MODEL_VERSION, features_train, label_drive_train, transform_pipeline_drive)

        features_test = load_dataset('features_win_prob_CFB_ot_test')[featureNames]
        label_drive_test = load_dataset('label_win_prob_CFB_ot_test').values.ravel()
        #label_drive_test = features_test.drive_outcome

        print(features_train.shape, features_test.shape)

        evaluate_multi_classification(model_drive, MODEL_VERSION, features_train, label_drive_train,
                                      features_test, label_drive_test)

        # train final model
        features = pd.concat((features_train, features_test), axis=0)
        label_drive = np.concatenate((label_drive_train, label_drive_test), axis=0) #features.drive_outcome

        model_drive = model_train(hyperparams, MODEL_VERSION, features[featureNames], label_drive,
                                  transform_pipeline_drive)

        mlflow.set_tags(mlflow_tags)

        mlflow.log_param('model_para', hyperparams)
        mlflow.log_param('model_version', MODEL_VERSION)

        mlflow.log_param('drive_model_para', hyperparams)
        mlflow.log_param('drive_model_version', MODEL_VERSION)

        mlflow.sklearn.log_model(model_drive, MODEL, registered_model_name=MODEL)

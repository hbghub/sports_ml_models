#!/usr/bin/env python

import argparse
import sys
from random import random
from statistics import mean
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

from src.utils import load_dataset, feature_names, num_fields_player, cat_fields_player, hyperparams, \
                        MODEL, MODEL_VERSION, model_train, evaluate_classification

mlflow.set_experiment("nba-player-props-prediction-models")

mlflow_tags = {
    'product': "Predictions",  # str or list[str]: fill in related products(s)
    'data': "Event",  # str or list[str]: fill in data(m) the model uses
    'sport': "Basketball",  # str or list[str]: fill in related sport(s)
    'short_description': "pre-game prediction models for nba players",  # str: fill in single sentence description of your model
    'documentation_url': "https://bitbucket.org/statsinc/nba-in-game-prediction-models/src/master/README.md"  # str or list[str]: fill in link(s) to documentation of your model
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


def model_fn(batch_size: int = 256):
    return mean([random() for _ in range(batch_size)])


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
        features_train = load_dataset('features_player_train')[feature_names]
        label_train = load_dataset('label_player_train').values.ravel()

        features_test = load_dataset('features_player_test')[feature_names]
        label_test = load_dataset('label_player_test').values.ravel()

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])

        # None-StandardScaler version
        transform_pipeline_player = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_fields_player),
            ('cat', OneHotEncoder(categories='auto'), cat_fields_player)
        ])

        model = model_train(hyperparams, MODEL_VERSION, features_train, label_train, transform_pipeline_player)

        evaluate_classification(model, MODEL_VERSION, features_train, label_train, features_test, label_test)

        # train final model
        features = pd.concat((features_train, features_test), axis=0)
        label = np.concatenate((label_train, label_test), axis=0)

        print("Training final model...")
        model = model_train(hyperparams, MODEL_VERSION, features, label, transform_pipeline_player)

        mlflow.set_tags(mlflow_tags)

        mlflow.log_param('model_version', MODEL_VERSION)
        mlflow.log_param('model', hyperparams)

        #mlflow.log_artifact('runtime/results/')
        mlflow.sklearn.log_model(model, MODEL, registered_model_name=MODEL)

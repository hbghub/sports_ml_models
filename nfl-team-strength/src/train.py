#!/usr/bin/env python

import argparse
import sys
from random import random
from statistics import mean

import mlflow

mlflow.set_experiment("nfl-team-strength")

mlflow_tags = {
    'product': "Predictions",  # str or list[str]: fill in related products(s)
    'data': "Box",  # str or list[str]: fill in data(m) the model uses
    'sport': "Football",  # str or list[str]: fill in related sport(s)
    'short_description': "model nfl team strength, such as elo rating",  # str: fill in single sentence description of your model
    'documentation_url': "https://bitbucket.org/statsinc/nfl-team-strength/src/master/README.md"  # str or list[str]: fill in link(s) to documentation of your model
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
        mlflow.set_tags(mlflow_tags)
        mlflow.log_params(params)
        for epoch in range(params.get('num_epochs', 10)):
            loss = model_fn()
            mlflow.log_metric("loss", loss)

        if params.get('model_dir') is not None:
            mlflow.log_artifacts(params['model_dir'])

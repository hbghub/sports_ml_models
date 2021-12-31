#!/usr/bin/env python

import mlflow
import argparse

parser = argparse.ArgumentParser(
    description="Command line arguments for registering a new version of a model."
)
parser.add_argument(
    "-r", "--run_id",
    required=True,
    type=str,
    help="mlflow run uuid of the model you wish to register"
)
args = parser.parse_args()

mlflow.register_model(f'runs:/{args.run_id}/', 'nfl-team-strength')

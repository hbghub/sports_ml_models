#!/usr/bin/env python

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import shutil
import argparse
import os


parser = argparse.ArgumentParser(
    description="Command line arguments for downloading a model."
)
parser.add_argument(
    "-v", "--version",
    required=True,
    type=int,
    help="version of the model to download"
)
parser.add_argument(
    "-md", "--model_dir",
    required=True,
    type=str,
    help="model directory"
)
args = parser.parse_args()

model_name = 'nfl-usage-models'

model_uri = f'models:/{model_name}/{args.version}'

temp_model_dir = _download_artifact_from_uri(model_uri)

shutil.move(temp_model_dir, os.path.join(args.model_dir, model_name, str(args.version)))

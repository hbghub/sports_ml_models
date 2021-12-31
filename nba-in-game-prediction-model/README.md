nba-in-game-prediction-models
==============================

in-game prediction models for nba teams and players

### Maintainer

* Name: bin hu
* Email: bin.hu@statsperform.global

### Project Organization

    ├── notebooks                <- Jupyter notebooks. 
    │   └── exploratory.ipynb    <- Sample Jupyter notebook which imports code in src/ directory.
    │
    ├── runtime                  <- Local runtime directory which can be mounted to the Docker container. 
    │   ├── checkpoints          <- Logs, model metadata and other files generated during training can be synced here.
    │   └── dataset              <- Datasets and their metadata/definitions can be synced here (likely from S3).
    │
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py          <- Makes src a Python module
    │   └── train.py             <- A sample training script with mlflow implemented.
    │
    ├── tests                    <- Test suite.
    │   ├── __init__.py
    │   └── test_train.py        <- A sample test file with some basic tests.
    │
    ├── .dockerignore            <- Standard set of things to ignore when creating Docker image.
    │
    ├── .gitignore               <- Standard set of things to ingore when making commits.
    │
    ├── Dockerfile               <- Boilerplate Python3.6 docker image definition. 
    │                               This will install the requirements and load all files from src/
    │
    ├── ludus.yaml               <- Template file for ludus entrypoints with sample definition for the train script. 
    │
    ├── README.md                <- The top-level README for developers using this project (aka this file).
    │
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.sh                 <- Script to setup git repo.

--------
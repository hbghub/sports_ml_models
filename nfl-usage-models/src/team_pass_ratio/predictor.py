import pandas as pd
import numpy as np
import mlflow.sklearn

import json, yaml

from utils import featureNames, predictProb, load_dataset

class PythonPredictor:

    def __init__(self, config):
        self.model_uri = config["model_uri"]
        print(f'Model URI: {self.model_uri}')
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, payload):

        # playerIds = payload['playerIds']
        # defenseTeamIds = payload['defenseTeamIds']

        game_df = pd.DataFrame(payload['records'])

        if predictProb:
            pred = self.model.predict_proba(game_df[featureNames])[:,1]
        else:
            pred = self.model.predict(game_df[featureNames])

        predictions = { k:v for k, v in zip(payload['teamid'], pred) }

        return predictions    # predicted value


def main():
    with open("cortex.yaml", 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config = content[0]['predictor']['config']

    # load data from local files 'runtime/datasets/features_...' into payload
    # in production, there may be a conversion between feed and the data frame format
    df = load_dataset('features_pass_ratio_test')
    df.sort_values(by=['season','week'], inplace=True)
    df = df[:3]

    teamId = df['teamid']
    records = df[featureNames]

    records = records.to_dict('records')

    payload = {}
    payload['records'] = records
    payload['teamid'] = teamId.tolist()

    with open('test.json','w') as outfile:
        json.dump(payload, outfile)

    #print(json.dumps(payload, indent=4))

    predictor = PythonPredictor(config)

    print(predictor.predict(payload))

    return 0


if __name__ == '__main__':
    exit(main())

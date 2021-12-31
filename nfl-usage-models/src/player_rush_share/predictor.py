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
        game_df = pd.DataFrame(payload['records'])

        playerids = game_df['playerid']
        teamids = game_df['teamid']

        game_df = game_df[featureNames]

        id = game_df.ytd_rushingShareByPositionRankAdj > 0
        game_df.loc[id, 'ytd_rushingShareAdj'] = game_df.ytd_rushingShareByPositionRankAdj[id]
        game_df.drop(columns=['ytd_rushingShareByPositionRankAdj'], inplace=True)

        if predictProb:
            pred = self.model.predict_proba(game_df)[:,1]
        else:
            pred = self.model.predict(game_df)

        # make sure non-negative prob.
        pred = [x if x >= 0.0 else 0.0 for x in pred ]

        # normalization of shares
        tmp = list(zip(pred, teamids))
        predictions = pd.DataFrame(tmp, columns=('pred', 'teamid'))
        predictions = predictions.groupby(by='teamid').transform(lambda x: x/x.sum())

        predictions = {k: v for k, v in zip(playerids, predictions.pred)}

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
    df = load_dataset('features_rush_share_test')

    df.sort_values(by=['season','week','teamid'], inplace=True)

    df = df[(df.week==3) & (df.season==2019)]
    #df = df[601:605]

    records = df[featureNames + ['playerid','teamid']]

    records = records.to_dict('records')

    payload = {}
    payload['records'] = records

    with open('test.json','w') as outfile:
        json.dump(payload, outfile)

    #print(json.dumps(payload, indent=4))

    predictor = PythonPredictor(config)

    print(predictor.predict(payload))

    return 0


if __name__ == '__main__':
    exit(main())

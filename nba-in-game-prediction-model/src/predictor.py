import pandas as pd
import numpy as np
import mlflow.sklearn

import json, yaml

from utils import feature_names, load_dataset_prediction #, predictProb

class PythonPredictor:

    def __init__(self, config):
        self.model_uri = config["model_uri"]
        print(f'Model URI: {self.model_uri}')
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, payload):
        features_test = pd.DataFrame(payload['records'])[feature_names]

        # predict distribution of
        pred_prob = self.model.predict_proba(features_test)

        pred = np.dot(pred_prob, np.arange(46))

        print(payload['player_id'])

        predictions = {payload['player_id'] : pred.tolist()}

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
    df = load_dataset_prediction('features_player_test')

    # # just pick one player, James Harden
    player_id = 395388
    df = df.loc[df.player_id==player_id,]

    records = df[feature_names]

    #records = records.to_dict('records')
    records = records.to_dict()

    payload = {}
    payload['records'] = records
    payload['player_id'] = int(player_id)

    with open('test.json','w') as outfile:
        json.dump(payload, outfile)
    # #print(json.dumps(payload, indent=4))

    predictor = PythonPredictor(config)

    import time
    start_time = time.time()
    re = predictor.predict(payload)
    print("--- %s seconds ---" % (time.time() - start_time))

    return 0


if __name__ == '__main__':
    exit(main())

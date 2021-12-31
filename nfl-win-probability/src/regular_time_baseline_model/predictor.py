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

        self.model_drive_uri = config["model_drive_uri"]
        print(f'Model Drive URI: {self.model_drive_uri}')
        self.model_drive = mlflow.sklearn.load_model(self.model_drive_uri)

    def predict(self, payload):
        features_test = pd.DataFrame([payload['records']])[featureNames]

        # 1st, fit the expected drive score
        drive_outcome_p = self.model_drive.predict_proba(features_test)
        expectedDriveScore = np.dot(drive_outcome_p, np.array([0, 1, 2, 3, 6, 7, 8, -2, -6]))
        features_test['expected_extra_score'] = features_test.drive_start_score_diff - features_test.score_diff + \
                                               expectedDriveScore
        features_test['adj_expected_score_diff'] = (features_test.expected_extra_score + features_test.score_diff) / \
                                                 np.power(features_test.remaining_game_time + 1, 0.5)

        # 2nd, predict final game results
        if predictProb:
            pred = self.model.predict_proba(features_test)[0,1]
        else:
            pred = self.model.predict(features_test[featureNames])

        print(payload['team_id'])

        predictions = {payload['team_id'] : pred}
        #predictions = { k:v for k, v in zip(payload['teamid'], pred) }

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
    df = load_dataset('features_win_prob_reg_baseline_test')
    df.sort_values(by=['season','week'], inplace=True)

    # just pick one play
    df = df.loc[(10),]

    teamId = df['offense_team']
    records = df[featureNames]

    #records = records.to_dict('records')
    records = records.to_dict()

    payload = {}
    payload['records'] = records
    payload['team_id'] = int(teamId) #.tolist()

    with open('test.json','w') as outfile:
        json.dump(payload, outfile)

    #print(json.dumps(payload, indent=4))

    predictor = PythonPredictor(config)

    import time
    start_time = time.time()
    re = predictor.predict(payload)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(re)

    return 0


if __name__ == '__main__':
    exit(main())

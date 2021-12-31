import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow.keras

import json, yaml

from utils import featureNames, predictProb, load_dataset

class PythonPredictor:

    def __init__(self, config):
        self.model_drive_uri = config["model_drive_uri"]
        print(f'Model drive URI: {self.model_drive_uri}')
        self.model_drive = mlflow.sklearn.load_model(self.model_drive_uri)

        self.model_ANN_uri = config["model_ANN_uri"]
        print(f'Model ANN URI: {self.model_ANN_uri}')
        self.model_ANN = mlflow.keras.load_model(self.model_ANN_uri)

        self.ANN_preprocessing_uri = config["ANN_preprocessing_uri"]
        print(f'ANN Preprocessing URI: {self.ANN_preprocessing_uri}')
        self.ANN_preprocessing = mlflow.sklearn.load_model(self.ANN_preprocessing_uri)

    def predict(self, payload):

        features = pd.DataFrame([payload['records']])[featureNames]

        # 1st, fit the expected drive score
        drive_outcome_p = self.model_drive.predict_proba(features)
        expectedDriveScore = np.dot(drive_outcome_p, np.array([0, 1, 2, 3, 6, 7, 8, -2, -6]))
        features['expected_extra_score'] = features.drive_start_score_diff - features.score_diff + \
                                               expectedDriveScore
        features['adj_expected_score_diff'] = (features.expected_extra_score + features.score_diff) / \
                                                 np.power(features.remaining_game_time + 1, 0.5)

        # 2nd, ANN model for win prob prediction
        features = self.ANN_preprocessing.transform(features)

        pred = self.model_ANN.predict(features)[0,0]

        # Note: it is very important to convert the np.float32 data type for deployment purpose!
        predictions = {payload['teamid'] : float(pred)}

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
    payload['teamid'] = int(teamId) #.tolist()

    with open('test.json','w') as outfile:
        json.dump(payload, outfile)

    predictor = PythonPredictor(config)

    import time
    start_time = time.time()
    re = predictor.predict(payload)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(re)

    return 0


if __name__ == '__main__':
    exit(main())

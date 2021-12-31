import pandas as pd
import numpy as np
import mlflow.sklearn

import json, yaml

from utils import featureNames, predictProb, load_dataset, SAMPLE_SIZE

class PythonPredictor:

    def __init__(self, config):
        self.model_uri = config["model_uri"]
        print(f'Model URI: {self.model_uri}')
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def createDriveOutcomeDistributionsAtSwitch(self, features):
        probs = {}

        row = features.iloc[0]

        # create a basic starting play for team 2
        feature_2 = row.copy()

        feature_2.yards_to_go = 10
        feature_2.yards_from_goal = 25

        # feature_2.offenseFavoritePoints = -feature.offenseFavoritePoints
        feature_2.remaining_offense_timeouts = row.remaining_defense_timeouts  # This feature is difficult to predict by the end of 1st play, unless we skip it!
        feature_2.remaining_defense_timeouts = row.remaining_offense_timeouts

        feature_2.down = 1
        feature_2.play_design = 0

        feature_2.offense_team = row.defense_team
        feature_2.defense_team = row.offense_team

        for score_diff in (-8, -7, -6, -3, 0):

            feature_2.score_diff = score_diff
            feature_2.offense_score = row.offense_score + score_diff
            feature_2.adj_score_diff = feature_2.score_diff / np.power(feature_2.remaining_game_time + 1, 0.5)

            play_2_prob = self.model.predict_proba(feature_2.to_frame().transpose())[0, :]
            probs[score_diff] = play_2_prob

        return probs


    def updateTeamScoreByDriveOutcome(self, re, home_team, offense_team, home_team_score, away_team_score):
        if re in (3,):  # field goal
            if home_team == offense_team:
                home_team_score = home_team_score + 3
            else:
                away_team_score = away_team_score + 3
        elif re in [4]:  # touch-down
            if home_team == offense_team:
                home_team_score = home_team_score + 6
            else:
                away_team_score = away_team_score + 6
        elif re in [5]:  # touch-down
            if home_team == offense_team:
                home_team_score = home_team_score + 7
            else:
                away_team_score = away_team_score + 7
        elif re in [6]:  # touch-down
            if home_team == offense_team:
                home_team_score = home_team_score + 8
            else:
                away_team_score = away_team_score + 8
        elif re != 0:
            print("Warning: un-tracked drive-outcome ", re)

        return (home_team_score, away_team_score)


    # this is for team 1
    def simulatedOutcome_1(self, home_team, offense_team, defense_team, home_team_score, away_team_score, play_1_prob,
                           play_2_probs):

        re = np.random.multinomial(n=1, pvals=play_1_prob)
        re = np.where(re)[0][0]

        if re in [1, 2, 7, 8]:  # safety or defense TD

            if home_team == offense_team:
                away_team_score += 2
                # print('team 1 turn over lose!')
                return (False, home_team_score, away_team_score)
            else:
                home_team_score += 2
                # print('team 2 turn over lose!')
                return (True, home_team_score, away_team_score)
        else:
            home_team_score, away_team_score = self.updateTeamScoreByDriveOutcome(re, home_team, offense_team,
                                                                                  home_team_score, away_team_score)

            if home_team == offense_team:
                scoreDiff = away_team_score - home_team_score
            else:
                scoreDiff = home_team_score - away_team_score

            play_2_prob = play_2_probs[scoreDiff]

            return self.simulatedOutcome_2(home_team, defense_team, home_team_score, away_team_score, play_2_prob)


    # this is for team 2
    def simulatedOutcome_2(self, home_team, offense_team, home_team_score, away_team_score, play_2_prob):

        re = np.random.multinomial(n=1, pvals=play_2_prob)
        re = np.where(re)[0][0]

        if re in [1, 2, 7, 8]:  # safety or defense TD
            if home_team == offense_team:
                away_team_score += 2
                return (False, home_team_score, away_team_score)
            else:
                home_team_score += 2
                return (True, home_team_score, away_team_score)
        else:
            home_team_score, away_team_score = self.updateTeamScoreByDriveOutcome(re, home_team, offense_team,
                                                                                  home_team_score, away_team_score)

            if home_team_score > away_team_score:
                return (True, home_team_score, away_team_score)
            else:
                return (False, home_team_score, away_team_score)


    # function to simulate outcomes of OT
    # input: features for each play of one game
    def OvertimeWinProbabilityCalculation_CFB(self, features):

        #for index, row in features.iterrows():
        row = features.iloc[0]

        offense_team = row.offense_team
        defense_team = row.defense_team
        start_team = row.period_start_offense_team
        home_team = row.home_team
        start_score_diff = row.drive_start_score_diff

        home_team_wins = 0
        home_team_score = 0
        away_team_score = 0

        if offense_team == start_team:
            play_1_prob = self.model.predict_proba(features)[0, :]
            play_2_probs = self.createDriveOutcomeDistributionsAtSwitch(features)

            for i in range(SAMPLE_SIZE):
                re = self.simulatedOutcome_1(home_team, offense_team, defense_team, home_team_score, away_team_score,
                                        play_1_prob, play_2_probs)

                if re[0]:
                    home_team_wins += 1
                elif re[1] == re[2]:
                    home_team_wins += 0.5
                    # print(team_1, team_2, team_1_score, team_2_score, re[1], re[2])

        else:
            if offense_team == home_team:
                home_team_score = start_score_diff
            else:
                away_team_score = start_score_diff

            # play_2_prob now needs to be conditional on current score difference!
            play_2_prob = self.model.predict_proba(features)[0, :]
            #print(home_team, offense_team, home_team_score, away_team_score, play_2_prob)

            for i in range(SAMPLE_SIZE):
                re = self.simulatedOutcome_2(home_team, offense_team, home_team_score, away_team_score,
                                        play_2_prob)

                if re[0]:
                    home_team_wins += 1
                elif re[1] == re[2]:
                    home_team_wins += 0.5
                    # print(team_1, team_2, team_1_score, team_2_score, re[1], re[2])

        # This part is to address 2-point conversion after 4 periods past in OT

        wp = np.sum(home_team_wins) / float(SAMPLE_SIZE)

        return wp


    def predict(self, payload):
        play = pd.DataFrame(payload['records'], index=[0])

        np.random.seed(0)

        pred = self.OvertimeWinProbabilityCalculation_CFB(play)

        return pred   # predicted value


def main():

    with open("cortex.yaml", 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config = content[0]['predictor']['config']

    # load data from local files 'runtime/datasets/features_...' into payload
    # in production, there may be a conversion between feed and the data frame format
    df = load_dataset('features_win_prob_CFB_ot_train')

    df.sort_values(by=['season','week'], inplace=True)

    # just pick one play
    id = (df.game_code == 1526113) & (df.period > 4)
    df = df[id]

    df = df.iloc[45,]

    teamId = df['offense_team']
    records = df[featureNames]
    records = records.to_dict()

    payload = {}
    payload['records'] = records

    with open('test.json','w') as outfile:
        json.dump(payload, outfile)
    #print(json.dumps(payload, indent=4))

    predictor = PythonPredictor(config)

    import time
    start_time = time.time()
    re = predictor.predict(payload)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(teamId, re)

    return 0


if __name__ == '__main__':
    exit(main())

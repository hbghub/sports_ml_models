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

        self.model_team_drive_uri = config["model_team_drive_uri"]
        print(f'Model team drive URI: {self.model_team_drive_uri}')
        self.model_team_drive = mlflow.sklearn.load_model(self.model_team_drive_uri)

    # this is for the 1st drive of OT, starting with team_1
    def simulatedOutcome_1(self, team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob, play_prob):
        re = np.random.multinomial(n=1, pvals=play_prob)
        re = np.where(re)[0][0]

        if re in [4, 5, 6]:  # touch-down
            team_1_score = team_1_score + 6
            return (True, team_1_score, team_2_score)

        elif re in (3,):  # field goal
            team_1_score = team_1_score + 3
            return self.simulatedOutcome_2(team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob, team_2_prob)

        elif re in (0,):  # no score
            return self.simulatedOutcome_2(team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob, team_2_prob)

        else:  # safety or defense TD
            team_2_score += 2
            return (False, team_1_score, team_2_score)

    # this is for the 2nd drive of OT, starting with team_2
    def simulatedOutcome_2(self, team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob, play_prob):

        re = np.random.multinomial(n=1, pvals=play_prob)
        re = np.where(re)[0][0]

        if re in (4, 5, 6):  # touch-down
            team_2_score = team_2_score + 6
            return (False, team_1_score, team_2_score)

        elif re in (3,):  # field goal
            team_2_score = team_2_score + 3
            if team_2_score > team_1_score:
                # print('#2: team 2 field goal win the game')
                return (False, team_1_score, team_2_score)
            elif team_2_score == team_1_score:  # move to restart, 3rd drive
                # print("restart after team 2 field goal")
                return self.simulatedOutcome_3(team_1, team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob,
                                          team_1_prob)
            else:
                print("Error: team_2 field goal,", team_1_score, team_2_score)
                return (False, team_1_score, team_2_score)

        elif re in (0,):
            if team_1_score > team_2_score:
                # print('#1: team 1 field goal win the game', team_1_score, team_2_score)
                return (True, team_1_score, team_2_score)
            elif team_1_score == team_2_score:
                # print("restart after team 2 no score")
                return self.simulatedOutcome_3(team_1, team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob,
                                          team_1_prob)
            else:
                print("Error: team_2 no score")
                return (False, team_1_score, team_2_score)

        else:  # safety or defense TD
            team_1_score += 2
            # print("safety, team_1 wins,", team_1_score, team_2_score)
            return (True, team_1_score, team_2_score)

    # After the 1st round of one drive each team
    # team_1 means the team start offense from the beginning of OT
    # team_o and team_d mean offense team and defense team for each drive after the 1st round
    def simulatedOutcome_3(self, team_1, team_o, team_d, team_o_score, team_d_score, team_o_prob, team_d_prob, play_prob):

        re = np.random.multinomial(n=1, pvals=play_prob)
        re = np.where(re)[0][0]

        if re in (3, 4, 5, 6):  # touch-down or field goal
            team_o_score = team_o_score + 6
            if team_o == team_1:
                return (True, team_o_score, team_d_score)
            else:
                return (False, team_d_score, team_o_score)
        elif re == 0:
            return self.simulatedOutcome_3(team_1, team_d, team_o, team_d_score, team_o_score, team_d_prob, team_o_prob,
                                      team_d_prob)
        else:  # safety or defense TD
            team_d_score += 2
            if team_d == team_1:
                return (True, team_d_score, team_o_score)
            else:
                return (False, team_o_score, team_d_score)

    # function to simulate outcomes of OT
    # input: features for each play, starting offense team id, starting defense team id, driveOutcomes
    def OvertimeWinProbabilityCalculation(self, features, team_1, team_2, team_1_prob, team_2_prob, drive_prob):

        samples = 1000

        driveId = features.drive_id[0]
        offenseTeam = features.offense_team[0]

        team_1_wins = 0
        team_1_score = 0
        team_2_score = 0

        if driveId == 0:
            for i in range(samples):
                re = self.simulatedOutcome_1(team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob,
                                        drive_prob)
                if re[0]:
                    team_1_wins = team_1_wins + 1
                    # print(team_1, team_2, team_1_score, team_2_score, re[1], re[2])

        elif driveId == 1:
            team_2_score = features.score_diff[0]

            for i in range(samples):
                re = self.simulatedOutcome_2(team_1, team_2, team_1_score, team_2_score, team_1_prob, team_2_prob,
                                        drive_prob)
                if re[0]:
                    team_1_wins = team_1_wins + 1
                    # print(team_1, team_2, team_1_score, team_2_score, re[1], re[2])

        else:
            if offenseTeam == team_1:
                defenseTeam = team_2
                team_o_prob = team_1_prob
                team_d_prob = team_2_prob
            else:
                defenseTeam = team_1
                team_o_prob = team_2_prob
                team_d_prob = team_1_prob

            for i in range(samples):
                re = self.simulatedOutcome_3(team_1, offenseTeam, defenseTeam, 0, 0, team_o_prob, team_d_prob, drive_prob)
                if re[0]:
                    team_1_wins = team_1_wins + 1
                    # print(team_1, team_2, team_1_score, team_2_score, re[1], re[2])

        wp = np.sum(team_1_wins) / float(samples)

        if offenseTeam == team_2:
            wp = 1 - wp

        return wp

    def predict(self, payload):

        game_df = pd.DataFrame(payload['records'], index=[0])
        game_df_r = game_df.copy()
        game_df_r['offense_team'] = game_df.defense_team
        game_df_r['defense_team'] = game_df.offense_team

        pred_drive_prob = self.model.predict_proba(game_df[featureNames])[0]
        pred_team_prob = self.model_team_drive.predict_proba(game_df[featureNames])[0]
        pred_team_prob_r = self.model_team_drive.predict_proba(game_df_r[featureNames])[0]

        predictions = {payload['teamid']: pred_drive_prob[0].tolist()}

        # start the simulation process below!!!
        if (game_df.drive_id[0] % 2) == 0:
            team_1 = game_df.offense_team[0]
            team_2 = game_df.defense_team[0]
            team_1_prob = pred_team_prob
            team_2_prob = pred_team_prob_r
        else:
            team_1 = game_df.defense_team[0]
            team_2 = game_df.offense_team[0]
            team_1_prob = pred_team_prob_r
            team_2_prob = pred_team_prob

        re = self.OvertimeWinProbabilityCalculation(game_df, team_1, team_2, team_1_prob, team_2_prob, pred_drive_prob)

        print(re)

        return re   # predicted value


def main():

    with open("cortex.yaml", 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config = content[0]['predictor']['config']

    # load data from local files 'runtime/datasets/features_...' into payload
    # in production, there may be a conversion between feed and the data frame format
    df = load_dataset('features_win_prob_ot_baseline_test')
    df.sort_values(by=['season','week'], inplace=True)

    # just pick one play
    id = (df.game_code == 2142043) & (df.period > 4)
    df = df[id]
    df.drive_id = df.drive_id - df.drive_id.iloc[0]

    df = df.iloc[15,]

    teamId = df['offense_team']
    records = df[featureNames]

    records = records.to_dict()

    payload = {}
    payload['records'] = records
    payload['teamid'] = int(teamId)#.tolist()


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

import pytest
import pandas as pd
import numpy as np

from train import model_fn, get_command_line_arguments

from regular_time_DL_model import generate_dataset as nfl_generate_dataset
from overtime_baseline_model import generate_dataset as nfl_ot_generate_dataset
from CFB_regular_time_DL_model import generate_dataset as cfb_generate_dataset
from CFB_overtime_baseline_model import generate_dataset as cfb_ot_generate_dataset
import utils

@pytest.fixture(scope='class')
def class_scope():
    '''fixture at class level'''

@pytest.mark.usefixtures('class_scope')
class TestTrain:

    @pytest.fixture(autouse=True)
    def _request_example_pbp_data(self):
        self.pbp_jstr = '{"season":{"100098":2018,"100099":2018},"week":{"100098":6,"100099":6},"game_code":{"100098":2040985,"100099":2040985},' \
                        '"event_type_id":{"100098":1,"100099":1},"home_team":{"100098":351,"100099":351},"away_team":{"100098":354,"100099":354},' \
                        '"offense_team":{"100098":354,"100099":351},"defense_team":{"100098":351,"100099":354},"play_id":{"100098":64,"100099":66},' \
                        '"drive_id":{"100098":8,"100099":9},"period":{"100098":2,"100099":2},"seconds_remaining_in_period":{"100098":820.0,"100099":810.0},' \
                        '"game_time":{"100098":980.0,"100099":990.0},"remaining_game_time":{"100098":2620,"100099":2610},"half_game":{"100098":1,"100099":1},' \
                        '"down":{"100098":4,"100099":1},"yards_to_go":{"100098":4.0,"100099":10.0},"yards_from_goal":{"100098":74.0,"100099":78.0},' \
                        '"away_score":{"100098":14,"100099":14},"home_score":{"100098":3,"100099":3},"away_score_after":{"100098":14,"100099":14},' \
                        '"home_score_after":{"100098":3,"100099":3},"home_final_score":{"100098":13,"100099":13},"away_final_score":{"100098":34,"100099":34},' \
                        '"offense_score":{"100098":14,"100099":3},"play_type_id":{"100098":7,"100099":4},"play_name":{"100098":"Punt","100099":"Rush"},' \
                        '"play_design":{"100098":0,"100099":0},"offense_timeout":{"100098":0,"100099":0},"defense_timeout":{"100098":0,"100099":0}}'

        self.win_jstr = '{"season":{"100098":2018,"100099":2018},"week":{"100098":6,"100099":6},"game_code":{"100098":2040985,"100099":2040985},' \
                        '"event_type_id":{"100098":1,"100099":1},"home_team":{"100098":351,"100099":351},"away_team":{"100098":354,"100099":354},' \
                        '"offense_team":{"100098":354,"100099":351},"defense_team":{"100098":351,"100099":354},"play_id":{"100098":64,"100099":66},' \
                        '"drive_id":{"100098":8,"100099":9},"period":{"100098":2,"100099":2},"seconds_remaining_in_period":{"100098":820.0,"100099":810.0},' \
                        '"game_time":{"100098":980.0,"100099":990.0},"remaining_game_time":{"100098":2620,"100099":2610},"half_game":{"100098":1,"100099":1},' \
                        '"down":{"100098":4,"100099":1},"yards_to_go":{"100098":4.0,"100099":10.0},"yards_from_goal":{"100098":74.0,"100099":78.0},' \
                        '"away_score":{"100098":14,"100099":14},"home_score":{"100098":3,"100099":3},"away_score_after":{"100098":14,"100099":14},' \
                        '"home_score_after":{"100098":3,"100099":3},"home_final_score":{"100098":13,"100099":13},"away_final_score":{"100098":34,"100099":34},' \
                        '"offense_score":{"100098":14,"100099":3},"play_type_id":{"100098":7,"100099":4},"play_name":{"100098":"Punt","100099":"Rush"},' \
                        '"play_design":{"100098":0,"100099":0},"offense_timeout":{"100098":0,"100099":0},"defense_timeout":{"100098":0,"100099":0},' \
                        '"score_diff":{"100098":11.0,"100099":-11.0},"score_diff_after":{"100098":11.0,"100099":-11.0},"adj_score_diff":{"100098":0.2148617827,"100099":-0.2152728444},' \
                        '"home_timeout":{"100098":0,"100099":0},"away_timeout":{"100098":0,"100099":0},"remaining_home_timeouts":{"100098":3,"100099":3},' \
                        '"remaining_away_timeouts":{"100098":3,"100099":3},"remaining_offense_timeouts":{"100098":3.0,"100099":3.0},' \
                        '"remaining_defense_timeouts":{"100098":3.0,"100099":3.0},"drive_score_diff_change":{"100098":0.0,"100099":0.0},' \
                        '"drive_outcome":{"100098":0,"100099":0},"drive_start_score_diff":{"100098":11.0,"100099":-11.0},"offense_win":{"100098":true,"100099":false},' \
                        '"line_scope":{"100098":"Game","100099":"Game"},"line_type":{"100098":"current","100099":"current"},' \
                        '"favorite_points":{"100098":-1.5,"100099":-1.5},"favorite_team_id":{"100098":354,"100099":354},"offense_favorite_points":{"100098":-1.5,"100099":1.5}}'

        self.ot_jstr ='{"season":{"100082":2018,"100083":2018},"week":{"100082":6,"100083":6},"game_code":{"100082":2040985,"100083":2040985},' \
                      '"event_type_id":{"100082":1,"100083":1},"home_team":{"100082":351,"100083":351},"away_team":{"100082":354,"100083":354},' \
                      '"offense_team":{"100082":351,"100083":354},"defense_team":{"100082":354,"100083":351},"play_id":{"100082":41,"100083":43},' \
                      '"drive_id":{"100082":5,"100083":6},"period":{"100082":1,"100083":1},"seconds_remaining_in_period":{"100082":303.0,"100083":289.0},' \
                      '"game_time":{"100082":597.0,"100083":611.0},"remaining_game_time":{"100082":3003,"100083":2989},"half_game":{"100082":1,"100083":1},' \
                      '"down":{"100082":4,"100083":1},"yards_to_go":{"100082":3.0,"100083":10.0},"yards_from_goal":{"100082":84.0,"100083":44.0},' \
                      '"away_score":{"100082":7,"100083":7},"home_score":{"100082":3,"100083":3},"away_score_after":{"100082":7,"100083":7},' \
                      '"home_score_after":{"100082":3,"100083":3},"home_final_score":{"100082":13,"100083":13},"away_final_score":{"100082":34,"100083":34},' \
                      '"offense_score":{"100082":3,"100083":7},"play_type_id":{"100082":7,"100083":1},"play_name":{"100082":"Punt","100083":"Pass"},' \
                      '"play_design":{"100082":0,"100083":0},"offense_timeout":{"100082":0,"100083":0},"defense_timeout":{"100082":0,"100083":0},' \
                      '"score_diff":{"100082":-4.0,"100083":4.0},"score_diff_after":{"100082":-4.0,"100083":4.0},"adj_score_diff":{"100082":-0.0729810365,"100083":0.0731516956},' \
                      '"home_timeout":{"100082":0,"100083":0},"away_timeout":{"100082":0,"100083":0},"remaining_home_timeouts":{"100082":3,"100083":3},' \
                      '"remaining_away_timeouts":{"100082":3,"100083":3},"remaining_offense_timeouts":{"100082":3.0,"100083":3.0},' \
                      '"remaining_defense_timeouts":{"100082":3.0,"100083":3.0},"drive_score_diff_change":{"100082":0.0,"100083":7.0},' \
                      '"drive_outcome":{"100082":0,"100083":5},"drive_start_score_diff":{"100082":-4.0,"100083":4.0},"offense_win":{"100082":false,"100083":true},' \
                      '"line_scope":{"100082":"Game","100083":"Game"},"line_type":{"100082":"current","100083":"current"},"favorite_points":{"100082":-1.5,"100083":-1.5},' \
                      '"favorite_team_id":{"100082":354,"100083":354},"offense_favorite_points":{"100082":1.5,"100083":-1.5}}'

        self.cfb_win_jstr='{"season":{"10005":2015,"10006":2015},"week":{"10005":2,"10006":2},"game_code":{"10005":1528069,"10006":1528069},' \
                          '"event_type_id":{"10005":1,"10006":1},"home_team":{"10005":3430,"10006":3430},"away_team":{"10005":3452,"10006":3452},' \
                          '"offense_team":{"10005":3452,"10006":3430},"defense_team":{"10005":3430,"10006":3452},"play_id":{"10005":199,"10006":201},' \
                          '"drive_id":{"10005":25,"10006":26},"period":{"10005":4,"10006":4},"seconds_remaining_in_period":{"10005":900.0,"10006":889.0},' \
                          '"half_game":{"10005":2,"10006":2},"down":{"10005":4,"10006":1},"yards_to_go":{"10005":7.0,"10006":10.0},' \
                          '"yards_from_goal":{"10005":50.0,"10006":85.0},"away_score":{"10005":3,"10006":3},"home_score":{"10005":38,"10006":38},' \
                          '"away_score_after":{"10005":3,"10006":3},"home_score_after":{"10005":38,"10006":38},"home_final_score":{"10005":41,"10006":41},' \
                          '"away_final_score":{"10005":3,"10006":3},"offense_score":{"10005":3,"10006":38},"play_type_id":{"10005":7.0,"10006":4.0},' \
                          '"play_name":{"10005":"Punt","10006":"Rush"},"play_design":{"10005":0,"10006":0},"offense_timeout":{"10005":0,"10006":0},' \
                          '"defense_timeout":{"10005":0,"10006":0},"remaining_game_time":{"10005":900.0,"10006":889.0},"score_diff":{"10005":-35.0,"10006":35.0},' \
                          '"score_diff_after":{"10005":-35.0,"10006":35.0},"adj_score_diff":{"10005":-1.1660190581,"10006":1.1732026655},' \
                          '"home_timeout":{"10005":0,"10006":0},"away_timeout":{"10005":0,"10006":0},"remaining_home_timeouts":{"10005":3,"10006":3},' \
                          '"remaining_away_timeouts":{"10005":3,"10006":3},"remaining_offense_timeouts":{"10005":3.0,"10006":3.0},' \
                          '"remaining_defense_timeouts":{"10005":3.0,"10006":3.0},"drive_score_diff_change":{"10005":0.0,"10006":0.0},' \
                          '"drive_outcome":{"10005":0,"10006":0},"drive_start_score_diff":{"10005":-35.0,"10006":35.0},"period_start_offense_team":{"10005":3452,"10006":3452},' \
                          '"offense_win":{"10005":false,"10006":true}}'


        self.cfb_ot_jstr='{"season":{"100109":2015,"100110":2015},"week":{"100109":10,"100110":10},"game_code":{"100109":1526126,"100110":1526126},' \
                         '"event_type_id":{"100109":1,"100110":1},"home_team":{"100109":3480,"100110":3480},"away_team":{"100109":3482,"100110":3482},' \
                         '"offense_team":{"100109":3482,"100110":3480},"defense_team":{"100109":3480,"100110":3482},"play_id":{"100109":34,"100110":36},' \
                         '"drive_id":{"100109":4,"100110":5},"period":{"100109":1,"100110":1},"seconds_remaining_in_period":{"100109":310.0,"100110":295.0},' \
                         '"half_game":{"100109":1,"100110":1},"down":{"100109":4,"100110":1},"yards_to_go":{"100109":18.0,"100110":10.0},' \
                         '"yards_from_goal":{"100109":87.0,"100110":39.0},"away_score":{"100109":3,"100110":3},"home_score":{"100109":0,"100110":0},' \
                         '"away_score_after":{"100109":3,"100110":3},"home_score_after":{"100109":0,"100110":0},"home_final_score":{"100109":19,"100110":19},' \
                         '"away_final_score":{"100109":27,"100110":27},"offense_score":{"100109":3,"100110":0},"play_type_id":{"100109":7.0,"100110":2.0},' \
                         '"play_name":{"100109":"Punt","100110":"Incomplete Pass"},"play_design":{"100109":0,"100110":0},"offense_timeout":{"100109":0,"100110":0},' \
                         '"defense_timeout":{"100109":0,"100110":0},"remaining_game_time":{"100109":3010.0,"100110":2995.0},"score_diff":{"100109":3.0,"100110":-3.0},' \
                         '"score_diff_after":{"100109":3.0,"100110":-3.0},"adj_score_diff":{"100109":0.0546721153,"100110":-0.0548088071},' \
                         '"home_timeout":{"100109":0,"100110":0},"away_timeout":{"100109":0,"100110":0},"remaining_home_timeouts":{"100109":3,"100110":3},' \
                         '"remaining_away_timeouts":{"100109":3,"100110":3},"remaining_offense_timeouts":{"100109":3.0,"100110":3.0},' \
                         '"remaining_defense_timeouts":{"100109":3.0,"100110":3.0},"drive_score_diff_change":{"100109":0.0,"100110":3.0},' \
                         '"drive_outcome":{"100109":0,"100110":3},"drive_start_score_diff":{"100109":3.0,"100110":-3.0},"period_start_offense_team":{"100109":3482,"100110":3482},' \
                         '"offense_win":{"100109":true,"100110":false}}'

        self.end_year = 2019

    def test_nfl_win_probability(self):
        df = pd.read_json(self.pbp_jstr, orient='columns', dtype=False)

        #(1) test score_diff
        utils.calculate_score_diff(df)

        re = df[['score_diff','score_diff_after','adj_score_diff']]

        re_o = pd.read_json('{"score_diff":{"100098":11.0,"100099":-11.0},'
                            '"score_diff_after":{"100098":11.0,"100099":-11.0},'
                            '"adj_score_diff":{"100098":0.2148617827,"100099":-0.2152728444}}',
                            orient='columns', dtype=False)

        assert np.allclose(re, re_o, rtol=1.0e-8) and \
               re.index.equals(re_o.index) and \
               re.columns.equals(re_o.columns), 'test failed on utils.calculate_score_diff'

        #(2) test remaining_timeout for NFL
        utils.calculate_remaining_timeout(df)
        re = df[['remaining_offense_timeouts', 'remaining_defense_timeouts']]
        re_o = pd.read_json('{"remaining_offense_timeouts":{"100098":3.0,"100099":3.0},"remaining_defense_timeouts":{"100098":3.0,"100099":3.0}}',
                            orient='columns', dtype=False)

        assert re.equals(re_o), 'test failed on utils.calculate_remaining_timeout'

        #(3) test remaining_timeout for CFB
        # utils.calculate_remaining_timeout_cfb(df)
        # re = df[['remaining_offense_timeouts', 'remaining_defense_timeouts']]
        # re_o = pd.read_json('{"remaining_offense_timeouts":{"100098":3.0,"100099":3.0},"remaining_defense_timeouts":{"100098":3.0,"100099":3.0}}',
        #                     orient='columns', dtype=False)
        #
        # assert re.equals(re_o), 'test failed on utils.calculate_remaining_timeout'



        #(4) test drive_outcome
        df = utils.calculate_drive_outcome(df)
        re = df[['drive_score_diff_change','drive_outcome', 'drive_start_score_diff']]
        re_o = pd.read_json('{"drive_score_diff_change":{"0":0.0,"1":0.0},"drive_outcome":{"0":0,"1":0},'
                            '"drive_start_score_diff":{"0":11.0,"1":-11.0}}',
                            orient='columns', dtype=False)

        assert re.equals(re_o), 'test failed on utils.calculate_drive_outcome'




    def test_generate_features(self):
        #(1) nfl regular time generate_features
        win_df = pd.read_json(self.win_jstr, orient='columns', dtype=False, precise_float=True)
        features_train, features_test, label_train, label_test = nfl_generate_dataset.generate_features(win_df, self.end_year)
        features_o = pd.read_json('{"score_diff":{"100098":11.0,"100099":-11.0},"adj_score_diff":{"100098":0.2148617827,"100099":-0.2152728444},'
                                  '"yards_to_go":{"100098":4.0,"100099":10.0},"yards_from_goal":{"100098":74.0,"100099":78.0},'
                                  '"offense_favorite_points":{"100098":-1.5,"100099":1.5},"seconds_remaining_in_period":{"100098":820.0,"100099":810.0},'
                                  '"remaining_game_time":{"100098":2620,"100099":2610},"offense_score":{"100098":14,"100099":3},'
                                  '"remaining_offense_timeouts":{"100098":3.0,"100099":3.0},"remaining_defense_timeouts":{"100098":3.0,"100099":3.0},'
                                  '"event_type_id":{"100098":1,"100099":1},"period":{"100098":2,"100099":2},"down":{"100098":4,"100099":1},'
                                  '"play_design":{"100098":0,"100099":0},"drive_outcome":{"100098":0,"100099":0},"drive_start_score_diff":{"100098":11.0,"100099":-11.0},'
                                  '"season":{"100098":2018,"100099":2018},"week":{"100098":6,"100099":6},"offense_team":{"100098":354,"100099":351}}',
                                  orient='columns', dtype=False, precise_float=True)

        label_o = pd.Series({100098:True, 100099:False})

        #special handling in order to compare floats
        assert np.allclose(features_train, features_o, rtol=1.0e-8), "error in creating features_train for regular_time_DL_model"
        assert label_train.equals(label_o), "error in creating label_train for regular_time_DL_model"

        #(2) nfl overtime generate_features()
        ot_df = pd.read_json(self.ot_jstr, orient='columns', dtype=False, precise_float=True)
        features_train, features_test, label_train, label_test = nfl_ot_generate_dataset.generate_features(ot_df, self.end_year)
        features_o = pd.read_json('{"score_diff":{"100082":-4.0,"100083":4.0},"yards_to_go":{"100082":3.0,"100083":10.0},"yards_from_goal":{"100082":84.0,"100083":44.0},'
                                  '"offense_favorite_points":{"100082":1.5,"100083":-1.5},"remaining_offense_timeouts":{"100082":3.0,"100083":3.0},'
                                  '"remaining_defense_timeouts":{"100082":3.0,"100083":3.0},"event_type_id":{"100082":1,"100083":1},'
                                  '"down":{"100082":4,"100083":1},"play_design":{"100082":0,"100083":0},"offense_team":{"100082":351,"100083":354},'
                                  '"defense_team":{"100082":354,"100083":351},"season":{"100082":2018,"100083":2018},"week":{"100082":6,"100083":6},'
                                  '"game_code":{"100082":2040985,"100083":2040985},"period":{"100082":1,"100083":1},"drive_id":{"100082":5,"100083":6}}',
                                  orient='columns', dtype=False, precise_float=True)

        label_o = pd.Series({100082:0, 100083:5})

        #special handling in order to compare floats
        assert np.allclose(features_train, features_o, rtol=1.0e-8), "error in creating features_train for overtime_baseline_model"
        assert label_train.equals(label_o), "error in creating label_train for overtime_baseline_model"

        #(3) CFB regular time generate_features
        cfb_win_df = pd.read_json(self.cfb_win_jstr, orient='columns', dtype=False, precise_float=True)
        features_train, features_test, label_train, label_test = cfb_generate_dataset.generate_features(cfb_win_df,
                                                                                                        self.end_year)
        features_o = pd.read_json('{"score_diff":{"10005":-35.0,"10006":35.0},"adj_score_diff":{"10005":-1.1660190581,"10006":1.1732026655},'
                                  '"yards_to_go":{"10005":7.0,"10006":10.0},"yards_from_goal":{"10005":50.0,"10006":85.0},"seconds_remaining_in_period":{"10005":900.0,"10006":889.0},'
                                  '"remaining_game_time":{"10005":900.0,"10006":889.0},"offense_score":{"10005":3,"10006":38},'
                                  '"remaining_offense_timeouts":{"10005":3.0,"10006":3.0},"remaining_defense_timeouts":{"10005":3.0,"10006":3.0},'
                                  '"event_type_id":{"10005":1,"10006":1},"period":{"10005":4,"10006":4},"down":{"10005":4,"10006":1},'
                                  '"play_design":{"10005":0,"10006":0},"drive_outcome":{"10005":0,"10006":0},"drive_start_score_diff":{"10005":-35.0,"10006":35.0},'
                                  '"game_code":{"10005":1528069,"10006":1528069},"season":{"10005":2015,"10006":2015},"week":{"10005":2,"10006":2},'
                                  '"offense_team":{"10005":3452,"10006":3430}}',
            orient='columns', dtype=False, precise_float=True)

        label_o = pd.Series({10005: False, 10006: True})

        # special handling in order to compare floats
        assert np.allclose(features_train, features_o,
                           rtol=1.0e-8), "error in creating features_train for CFB_regular_time_DL_model"
        assert label_train.equals(label_o), "error in creating label_train for CFB_regular_time_DL_model"

        #(4) cfb overtime generate_features()
        cfb_ot_df = pd.read_json(self.cfb_ot_jstr, orient='columns', dtype=False, precise_float=True)
        features_train, features_test, label_train, label_test = cfb_ot_generate_dataset.generate_features(cfb_ot_df, self.end_year)
        features_o = pd.read_json('{"score_diff":{"100109":3.0,"100110":-3.0},"adj_score_diff":{"100109":0.0546721153,"100110":-0.0548088071},' \
                         '"yards_to_go":{"100109":18.0,"100110":10.0},"yards_from_goal":{"100109":87.0,"100110":39.0},"seconds_remaining_in_period":{"100109":310.0,"100110":295.0},' \
                         '"remaining_game_time":{"100109":3010.0,"100110":2995.0},"offense_score":{"100109":3,"100110":0},"remaining_offense_timeouts":{"100109":3.0,"100110":3.0},' \
                         '"remaining_defense_timeouts":{"100109":3.0,"100110":3.0},"event_type_id":{"100109":1,"100110":1},"period":{"100109":1,"100110":1},' \
                         '"down":{"100109":4,"100110":1},"play_design":{"100109":0,"100110":0},"game_code":{"100109":1526126,"100110":1526126},' \
                         '"drive_start_score_diff":{"100109":3.0,"100110":-3.0},"offense_team":{"100109":3482,"100110":3480},' \
                         '"defense_team":{"100109":3480,"100110":3482},"home_team":{"100109":3480,"100110":3480},"period_start_offense_team":{"100109":3482,"100110":3482},' \
                         '"season":{"100109":2015,"100110":2015},"week":{"100109":10,"100110":10}}',
                                  orient='columns', dtype=False, precise_float=True)

        label_o = pd.Series({100109:0, 100110:3})

        #special handling in order to compare floats
        assert np.allclose(features_train, features_o, rtol=1.0e-8), "error in creating features_train for CFB_overtime_baseline_model"
        assert label_train.equals(label_o), "error in creating label_train for CFB_overtime_baseline_model"

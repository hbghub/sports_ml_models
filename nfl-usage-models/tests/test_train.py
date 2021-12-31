import pandas as pd
import numpy as np
import pytest

#from train import model_fn, get_command_line_arguments

import utils
from src.player_target_share.generate_dataset import calculate_player_adjusted_target_share


class TestTrain:

    @pytest.fixture(autouse=True)
    def _request_example_pbp_data(self):
        self.passing_jstr='{"base_passingpercentage":{"534":0.5627208481,"535":0.5627208481,"536":0.5627208481},"ytd_totalPlays":{"534":null,"535":55.0,"536":113.0},' \
                        '"ytd_passingpercentage":{"534":null,"535":0.5818181818,"536":0.5575221239},"exp_totalPlays":{"534":null,"535":236.446142823,"536":135.275339186}}'

        self.rushing_jstr='{"season":{"16362":2019,"16363":2019,"16364":2019},"playerid":{"16362":498760,"16363":465752,"16364":749635},' \
                          '"ytd_onFieldTotalTruePassAttempts":{"16362":96.0,"16363":37.0,"16364":1.0},"base_onFieldTotalTruePassAttempts":{"16362":96.0,"16363":5.0,"16364":2.0},' \
                          '"ytd_targetShare_2":{"16362":0.0985221675,"16363":0.0337837838,"16364":0.0079365079},"base_targetShare":{"16362":0.0674486804,"16363":0.0051282051,"16364":0.0}}'

    def test_calculate_team_usage(self):
        df = pd.read_json(self.passing_jstr, orient='columns', dtype=False)

        utils.calculate_team_expected_passing(df)

        re = df[['reg_passingpercentage', 'exp_passingPlays']]

        re_o = pd.read_json('{"reg_passingpercentage":{"534":null,"535":0.5653143131,"536":0.561452045},'
                            '"exp_passingPlays":{"534":null,"535":133.6663888235,"536":75.9506158196}}',
                            orient='columns', dtype=False)

        assert np.allclose(re[1:], re_o[1:], rtol=1.0e-8) and \
               np.all(np.isnan(re[:1])) and \
               re.index.equals(re_o.index) and \
               re.columns.equals(re_o.columns), 'test failed on utils.calculate_ytd_passing_yards'

    def test_calculate_player_usage(self):
        df = pd.read_json(self.rushing_jstr, orient='columns', dtype=False)
        alpha = 5.0

        calculate_player_adjusted_target_share(alpha, df)

        re = df['ytd_targetShareAdj']

        re_o = pd.read_json('{"16362":0.093343253,"16363":0.0330296896,"16364":0.0056689342}',
                            typ='series')

        assert np.allclose(re.values, re_o.values, rtol=1.0e-8), 'test failed on calculate_player_adjusted_target_share'


    # def test_model_fn(self):
    #     loss = model_fn()
    #     assert isinstance(loss, float)
    #
    # def test_get_command_line_arguments_default(self):
    #     args = []
    #     actual = get_command_line_arguments(args)
    #     expected = {
    #         'batch_size': 256,
    #         'num_epochs': 10
    #     }
    #     print(actual)
    #     assert expected == actual
    #
    # def test_get_command_line_arguments_custom(self):
    #     args = ['--batch-size', '32', '--num-epochs', '100']
    #     actual = get_command_line_arguments(args)
    #     expected = {
    #         'batch_size': 32,
    #         'num_epochs': 100
    #     }
    #     print(actual)
    #     assert expected == actual
    #
    # def test_get_command_line_arguments_custom_with_list(self):
    #     args = [
    #         '--batch-size', '32',
    #         '--num-epochs', '100',
    #         '--hidden-units', '1', '2', '3', '4'
    #     ]
    #     actual = get_command_line_arguments(args)
    #     expected = {
    #         'batch_size': 32,
    #         'num_epochs': 100,
    #         'hidden_units': [1, 2, 3, 4]
    #     }
    #     print(actual)
    #     assert expected == actual
    #
    # def test_get_command_line_arguments_no_dict(self):
    #     args = []
    #     actual = get_command_line_arguments(args, return_dict=False)
    #     assert actual.batch_size == 256

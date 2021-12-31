

from train import model_fn, get_command_line_arguments


class TestTrain:

    def test_model_fn(self):
        loss = model_fn()
        assert isinstance(loss, float)

    def test_get_command_line_arguments_default(self):
        args = []
        actual = get_command_line_arguments(args)
        expected = {
            'batch_size': 256,
            'num_epochs': 10
        }
        print(actual)
        assert expected == actual

    def test_get_command_line_arguments_custom(self):
        args = ['--batch-size', '32', '--num-epochs', '100']
        actual = get_command_line_arguments(args)
        expected = {
            'batch_size': 32,
            'num_epochs': 100
        }
        print(actual)
        assert expected == actual

    def test_get_command_line_arguments_custom_with_list(self):
        args = [
            '--batch-size', '32',
            '--num-epochs', '100',
            '--hidden-units', '1', '2', '3', '4'
        ]
        actual = get_command_line_arguments(args)
        expected = {
            'batch_size': 32,
            'num_epochs': 100,
            'hidden_units': [1, 2, 3, 4]
        }
        print(actual)
        assert expected == actual

    def test_get_command_line_arguments_no_dict(self):
        args = []
        actual = get_command_line_arguments(args, return_dict=False)
        assert actual.batch_size == 256

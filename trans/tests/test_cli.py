"""Unit tests for CLI parser fragments."""
import argparse
import unittest

from trans import encoders, optimizers


class TestBooleanArguments(unittest.TestCase):

    def test_lstm_bidirectional_boolean_optional_action(self):
        parser = argparse.ArgumentParser()
        encoders.LSTMEncoder.add_args(parser)

        self.assertTrue(parser.parse_args([]).enc_bidirectional)
        self.assertFalse(parser.parse_args(["--no-enc-bidirectional"]).enc_bidirectional)

    def test_optimizer_boolean_optional_actions(self):
        parser = argparse.ArgumentParser()
        optimizers.Adam.add_args(parser)
        optimizers.ReduceLROnPlateau.add_args(parser)

        args = parser.parse_args(["--amsgrad", "--verbose"])
        self.assertTrue(args.amsgrad)
        self.assertTrue(args.verbose)

        args = parser.parse_args(["--no-amsgrad", "--no-verbose"])
        self.assertFalse(args.amsgrad)
        self.assertFalse(args.verbose)


if __name__ == "__main__":
    unittest.main()

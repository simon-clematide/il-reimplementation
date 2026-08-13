"""Unit tests for CLI parser fragments."""
import argparse
import subprocess
import sys
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

    def test_train_help_includes_selected_scheduler_defaults(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trans.train",
                "--scheduler",
                "reduce_on_plateau",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        normalized_stdout = " ".join(result.stdout.split())
        self.assertIn("--lrs-patience", result.stdout)
        self.assertIn("(default: 2)", normalized_stdout)
        self.assertIn("--factor", result.stdout)
        self.assertIn("(default: 0.5)", normalized_stdout)

    def test_train_help_includes_default_component_defaults(self):
        result = subprocess.run(
            [sys.executable, "-m", "trans.train", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--enc-hidden-dim", result.stdout)
        self.assertIn("(default: 200)", result.stdout)
        self.assertIn("--enc-output-dropout", result.stdout)
        self.assertIn("--enc-output-dropout-type", result.stdout)
        self.assertIn("--rho", result.stdout)
        self.assertIn("(default: 0.9)", result.stdout)

    def test_train_help_includes_selected_encoder_and_optimizer_defaults(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trans.train",
                "--enc-type",
                "transformer",
                "--optimizer",
                "adam",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--enc-nhead", result.stdout)
        self.assertIn("(default: 4)", result.stdout)
        self.assertIn("--betas", result.stdout)
        self.assertIn("(default: (0.9, 0.999))", result.stdout)


if __name__ == "__main__":
    unittest.main()

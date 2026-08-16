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
        self.assertIn("--source-separator", result.stdout)
        self.assertIn("--target-separator", result.stdout)
        self.assertIn("--output-feedback-dim", result.stdout)
        self.assertIn("--expert-temperature", result.stdout)
        self.assertIn("--expert-loss", result.stdout)
        self.assertIn("--focal-gamma", result.stdout)
        self.assertIn("--focal-start", result.stdout)
        self.assertIn("--focal-ramp", result.stdout)
        self.assertIn("--expert-margin", result.stdout)
        self.assertIn("--contrastive-negative", result.stdout)
        self.assertIn("--critic-model-action", result.stdout)
        self.assertIn("--critic-augment-model-action", result.stdout)
        self.assertIn("--reload-best-on-lr-reduction", result.stdout)
        self.assertIn("--rollin-prob", result.stdout)
        self.assertIn("--rollin-start", result.stdout)
        self.assertIn("--rollin-refresh", result.stdout)
        self.assertIn("--rollin-policy", result.stdout)
        self.assertIn("--rollin-seed", result.stdout)
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

    def test_focal_marginal_rejects_positive_expert_temperature(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trans.train",
                "--dev",
                "dev.tsv",
                "--output",
                "out",
                "--expert-loss",
                "focal_marginal",
                "--expert-temperature",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "--expert-loss=focal_marginal requires --expert-temperature=0",
            result.stderr,
        )

    def test_contrastive_rejects_positive_expert_temperature(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trans.train",
                "--dev",
                "dev.tsv",
                "--output",
                "out",
                "--expert-loss",
                "contrastive",
                "--expert-temperature",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "--expert-loss=contrastive requires --expert-temperature=0",
            result.stderr,
        )

    def test_reload_best_on_lr_reduction_requires_reduce_on_plateau(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trans.train",
                "--dev",
                "dev.tsv",
                "--output",
                "out",
                "--reload-best-on-lr-reduction",
            ],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "--reload-best-on-lr-reduction requires --scheduler=reduce_on_plateau",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()

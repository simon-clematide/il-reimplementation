"""Unit tests for grid-search command construction."""
import unittest
from unittest import mock

from trans import grid_search


class TestGridSearchCommands(unittest.TestCase):

    def test_build_option_args_omits_false_booleans(self):
        args = grid_search.build_option_args({
            "nfd": False,
            "beam-width": 4,
            "enc-type": "lstm",
        })

        self.assertEqual(["--beam-width", "4", "--enc-type", "lstm"], args)

    def test_build_option_args_emits_true_booleans(self):
        args = grid_search.build_option_args({"nfd": True})

        self.assertEqual(["--nfd"], args)

    def test_build_ensemble_command_keeps_executable_separate(self):
        command = grid_search.build_ensemble_command(
            gold="dev.tsv",
            systems=["a.pred", "b.pred"],
            output="ensemble",
        )

        self.assertEqual(
            [
                "trans-ensemble",
                "--gold", "dev.tsv",
                "--systems", "a.pred", "b.pred",
                "--output", "ensemble",
            ],
            command,
        )

    def test_run_ensemble_raises_on_failure(self):
        process = mock.Mock()
        process.wait.return_value = 7
        with mock.patch.object(grid_search.subprocess, "Popen", return_value=process):
            with self.assertRaises(RuntimeError):
                grid_search.run_ensemble("gold", ["system"], "output")


if __name__ == "__main__":
    unittest.main()

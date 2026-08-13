"""Unit tests for grid-search command construction."""
import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

from trans import grid_search


class TestGridSearchCommands(unittest.TestCase):

    def tearDown(self):
        grid_search.ACTIVE_PROCESSES.clear()

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

    def test_grid_search_combinations_returns_argument_lists_and_metadata(self):
        args_list, comb_dict = grid_search.grid_search_combinations({
            "beam-width": [0, 5],
            "nfd": [False, True],
            "sed-params": {"ita": "ita.pkl"},
        })

        self.assertEqual(4, len(args_list))
        self.assertEqual(
            {"beam-width": 0, "nfd": False},
            comb_dict[1],
        )
        self.assertEqual(["--beam-width", "0"], args_list[0])

    def test_beam_width_from_combination_uses_configured_value(self):
        self.assertIsNone(grid_search.beam_width_from_combination({}))
        self.assertIsNone(grid_search.beam_width_from_combination({"beam-width": 0}))
        self.assertIsNone(grid_search.beam_width_from_combination({"beam-width": "0"}))
        self.assertEqual(
            "beam5",
            grid_search.beam_width_from_combination({"beam-width": 5}),
        )
        with self.assertRaisesRegex(ValueError, "beam-width"):
            grid_search.beam_width_from_combination({"beam-width": -1})
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            grid_search.beam_width_from_combination({"beam-width": "foo"})
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            grid_search.beam_width_from_combination({"beam-width": 2.5})

    def test_default_parallel_jobs_depends_on_configured_device(self):
        self.assertEqual(
            30,
            grid_search.default_parallel_jobs({
                "grids": {"grid": {"device": ["CPU"]}},
            }),
        )
        self.assertEqual(
            4,
            grid_search.default_parallel_jobs({
                "grids": {"grid": {"device": ["cpu", "mps"]}},
            }),
        )

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

    def test_validate_parallel_jobs_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            grid_search.validate_parallel_jobs(0)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            grid_search.validate_parallel_jobs(-1)

    def test_process_manager_reports_context_on_failure(self):
        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 9

        manager = grid_search.ProcessManager(parallel_jobs=1, poll_interval=0)
        with mock.patch.object(grid_search.subprocess, "Popen", return_value=process):
            manager.start(
                ["trans-train", "--output", "out"],
                output="out",
                grid="grid-a",
                language="ita",
                combination=3,
                run=2,
            )

        with self.assertRaisesRegex(
                RuntimeError,
                "return code 9: grid=grid-a language=ita combination=3 run=2"):
            manager.wait_for_slot()

    def test_process_manager_terminates_and_waits_for_active_processes(self):
        process = mock.Mock()
        process.poll.side_effect = [None, 0, 0]

        manager = grid_search.ProcessManager(parallel_jobs=2, poll_interval=0)
        running = grid_search.RunningProcess(
            process=process,
            command=["trans-train"],
            output="out",
            grid="grid-a",
            language="ita",
            combination=1,
            run=1,
        )
        manager.processes.append(running)
        grid_search.ACTIVE_PROCESSES.append(running)

        manager.terminate_all()

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5.)
        self.assertEqual([], manager.processes)
        self.assertEqual([], grid_search.ACTIVE_PROCESSES)

    def test_process_manager_kills_after_terminate_timeout(self):
        process = mock.Mock()
        process.poll.side_effect = [None, 0, 0]
        process.wait.side_effect = [
            grid_search.subprocess.TimeoutExpired(["trans-train"], timeout=5.),
            0,
        ]

        manager = grid_search.ProcessManager(parallel_jobs=2, poll_interval=0)
        running = grid_search.RunningProcess(
            process=process,
            command=["trans-train"],
            output="out",
            grid="grid-a",
            language="ita",
            combination=1,
            run=1,
        )
        manager.processes.append(running)
        grid_search.ACTIVE_PROCESSES.append(running)

        manager.terminate_all()

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(2, process.wait.call_count)
        self.assertEqual([], manager.processes)

    def test_last_value_from_file_reports_empty_or_malformed_eval_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.eval")
            malformed = os.path.join(tmp, "malformed.eval")
            with open(empty, "w"):
                pass
            with open(malformed, "w") as f:
                f.write("accuracy nope\n")

            with self.assertRaisesRegex(ValueError, "Empty evaluation file"):
                grid_search.last_value_from_file(empty)
            with self.assertRaisesRegex(ValueError, "Could not parse"):
                grid_search.last_value_from_file(malformed)

    def test_write_to_results_file_uses_per_result_beam_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_file = os.path.join(tmp, "results.txt")
            grid_search.write_to_results_file(results_file, [
                {
                    "c_dir": "1",
                    "beam_width": "beam2",
                    "dev_greedy": 0.2,
                    "dev_beam": 0.25,
                    "test_greedy": None,
                    "test_beam": None,
                },
                {
                    "c_dir": "2",
                    "beam_width": "beam5",
                    "dev_greedy": 0.3,
                    "dev_beam": 0.35,
                    "test_greedy": None,
                    "test_beam": None,
                },
            ])

            with open(results_file) as f:
                contents = f.read()

        self.assertIn("beam2: 0.25", contents)
        self.assertIn("beam5: 0.35", contents)

    def test_write_to_results_file_prints_zero_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_file = os.path.join(tmp, "results.txt")
            grid_search.write_to_results_file(results_file, [
                {
                    "c_dir": "1",
                    "beam_width": "beam2",
                    "dev_greedy": 0.0,
                    "dev_beam": 0.0,
                    "test_greedy": 0.0,
                    "test_beam": 0.0,
                },
            ])

            with open(results_file) as f:
                contents = f.read()

        self.assertIn("greedy: 0.0", contents)
        self.assertIn("beam2: 0.0", contents)
        self.assertIn("test", contents)

    def test_require_files_reports_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "existing.eval")
            missing = os.path.join(tmp, "missing.eval")
            with open(existing, "w") as f:
                f.write("accuracy 0.5\n")

            with self.assertRaisesRegex(FileNotFoundError, "missing.eval"):
                grid_search.require_files([existing, missing])

    def test_validate_config_checks_required_input_files_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "runs_per_model": 1,
                "data": {
                    "path": tmp,
                    "pattern": "LANG.SPLIT.tsv",
                    "languages": ["ita"],
                },
                "grids": {"grid": {"beam-width": [0]}},
            }
            with open(os.path.join(tmp, "ita.dev.tsv"), "w") as f:
                f.write("a\ta\n")

            with self.assertRaisesRegex(FileNotFoundError, "ita.train.tsv"):
                try:
                    grid_search.validate_config(config)
                except FileNotFoundError as exc:
                    self.assertEqual(1, str(exc).count("ita.train.tsv"))
                    raise

    def test_validate_config_rejects_unknown_language_specific_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split in ("train", "dev"):
                with open(os.path.join(tmp, f"ita.{split}.tsv"), "w") as f:
                    f.write("a\ta\n")
            config = {
                "runs_per_model": 1,
                "data": {
                    "path": tmp,
                    "pattern": "LANG.SPLIT.tsv",
                    "languages": ["ita"],
                },
                "grids": {
                    "grid": {
                        "beam-width": [0],
                        "precomputed-train": {"tia": "typo.pkl"},
                    },
                },
            }

            with self.assertRaisesRegex(ValueError, "unknown languages"):
                grid_search.validate_config(config)

    def test_main_evaluation_ignores_ensemble_directories_in_run_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            output = os.path.join(tmp, "output")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            for split in ("train", "dev"):
                with open(os.path.join(data_dir, f"ita.{split}.tsv"), "w") as f:
                    f.write("a\ta\n")
            with open(config_path, "w") as f:
                json.dump({
                    "runs_per_model": 2,
                    "data": {
                        "path": data_dir,
                        "pattern": "LANG.SPLIT.tsv",
                        "languages": ["ita"],
                    },
                    "grids": {
                        "grid": {
                            "beam-width": [3],
                        },
                    },
                }, f)

            combo_dir = os.path.join(output, "grid", "ita", "1")
            for run_name, score in (("1.1", "0.2"), ("1.2", "0.4")):
                run_dir = os.path.join(combo_dir, run_name)
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, "dev_greedy.eval"), "w") as f:
                    f.write(f"accuracy {score}\n")
                with open(os.path.join(run_dir, "dev_beam3.eval"), "w") as f:
                    f.write(f"accuracy {score}\n")
            os.makedirs(os.path.join(combo_dir, "greedy_ensemble"))

            process = mock.Mock()
            process.poll.return_value = 0
            process.returncode = 0
            args = argparse.Namespace(
                config=config_path,
                output=output,
                parallel_jobs=1,
                ensemble=False,
            )

            with mock.patch.object(grid_search.subprocess, "Popen", return_value=process):
                grid_search.main(args)

            with open(os.path.join(output, "grid", "ita", "results.txt")) as f:
                contents = f.read()

        self.assertIn("greedy: 0.3", contents)
        self.assertIn("beam3: 0.3", contents)

    def test_main_evaluation_requires_test_artifacts_when_test_data_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            output = os.path.join(tmp, "output")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            for split in ("train", "dev"):
                with open(os.path.join(data_dir, f"ita.{split}.tsv"), "w") as f:
                    f.write("a\ta\n")
            with open(os.path.join(data_dir, "ita.test.tsv"), "w") as f:
                f.write("a\ta\n")
            with open(config_path, "w") as f:
                json.dump({
                    "runs_per_model": 1,
                    "data": {
                        "path": data_dir,
                        "pattern": "LANG.SPLIT.tsv",
                        "languages": ["ita"],
                    },
                    "grids": {
                        "grid": {
                            "beam-width": [0],
                        },
                    },
                }, f)

            run_dir = os.path.join(output, "grid", "ita", "1", "1.1")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "dev_greedy.eval"), "w") as f:
                f.write("accuracy 0.2\n")

            process = mock.Mock()
            process.poll.return_value = 0
            process.returncode = 0
            args = argparse.Namespace(
                config=config_path,
                output=output,
                parallel_jobs=1,
                ensemble=False,
            )

            with mock.patch.object(grid_search.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(FileNotFoundError, "test_greedy.eval"):
                    grid_search.main(args)


if __name__ == "__main__":
    unittest.main()

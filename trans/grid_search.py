"""CLI for performing grid search."""

import argparse
import dataclasses
import json
import subprocess
import os
import itertools
import math
import time
import atexit
from collections.abc import Sequence
from typing import Any, Optional, List


ACTIVE_PROCESSES = []
LANGUAGE_SPECIFIC_PARAMETERS = {
    "sed-params",
    "precomputed-train",
    "vocabulary",
}
TRUE_DEFAULT_BOOLEAN_PARAMETERS = {
    "enc-bidirectional",
}
CPU_PARALLEL_JOBS_DEFAULT = 30
ACCELERATOR_PARALLEL_JOBS_DEFAULT = 4


def cleanup():
    terminate_processes(ACTIVE_PROCESSES)


atexit.register(cleanup)


@dataclasses.dataclass
class RunningProcess:
    process: subprocess.Popen
    command: List[str]
    output: str
    grid: str
    language: str
    combination: int
    run: int

    def failure_message(self) -> str:
        return (
            f"Training command failed with return code {self.process.returncode}: "
            f"grid={self.grid} language={self.language} "
            f"combination={self.combination} run={self.run} "
            f"output={self.output} command={' '.join(self.command)}"
        )


def validate_parallel_jobs(parallel_jobs: int) -> None:
    if parallel_jobs < 1:
        raise ValueError(f"--parallel-jobs must be at least 1: {parallel_jobs}.")


def default_parallel_jobs(config_dict: dict) -> int:
    for grid_config in config_dict["grids"].values():
        for device in get_list(grid_config.get("device", "cpu")):
            if str(device).lower() != "cpu":
                return ACCELERATOR_PARALLEL_JOBS_DEFAULT
    return CPU_PARALLEL_JOBS_DEFAULT


def terminate_processes(processes: List[RunningProcess], timeout: float = 5.) -> None:
    for running in list(processes):
        process = running.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    processes[:] = [running for running in processes if running.process.poll() is None]


class ProcessManager:
    def __init__(self, parallel_jobs: int, poll_interval: float = 5.) -> None:
        validate_parallel_jobs(parallel_jobs)
        self.parallel_jobs = parallel_jobs
        self.poll_interval = poll_interval
        self.processes: List[RunningProcess] = []

    def start(self, command: List[str], *, output: str, grid: str,
              language: str, combination: int, run: int) -> RunningProcess:
        process = subprocess.Popen(command, bufsize=0)
        running = RunningProcess(
            process=process,
            command=command,
            output=output,
            grid=grid,
            language=language,
            combination=combination,
            run=run,
        )
        self.processes.append(running)
        ACTIVE_PROCESSES.append(running)
        return running

    def poll_finished(self) -> None:
        finished = [
            running for running in self.processes
            if running.process.poll() is not None
        ]
        failures = [
            running for running in finished
            if running.process.returncode != 0
        ]
        self.processes = [
            running for running in self.processes
            if running.process.poll() is None
        ]
        ACTIVE_PROCESSES[:] = [
            running for running in ACTIVE_PROCESSES
            if running.process.poll() is None
        ]
        if failures:
            raise RuntimeError(failures[0].failure_message())

    def wait_for_slot(self) -> None:
        while len(self.processes) >= self.parallel_jobs:
            self.poll_finished()
            if len(self.processes) >= self.parallel_jobs:
                time.sleep(self.poll_interval)

    def wait_all(self) -> None:
        while self.processes:
            self.poll_finished()
            if self.processes:
                time.sleep(self.poll_interval)

    def terminate_all(self) -> None:
        terminate_processes(self.processes)
        ACTIVE_PROCESSES[:] = [
            running for running in ACTIVE_PROCESSES
            if running.process.poll() is None
        ]


def last_value_from_file(file_path: str, t=float):
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Empty evaluation file: {file_path}")
        try:
            return t(lines[-1].split()[-1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Could not parse evaluation result from {file_path}: "
                f"{lines[-1]!r}") from exc


def summarize_scores(scores: List[float]) -> tuple:
    if not scores:
        raise ValueError("Cannot summarize empty score list.")
    mean = sum(scores) / len(scores)
    if len(scores) == 1:
        return mean, 0.
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return mean, math.sqrt(variance)


def format_score_summary(scores: List[float]) -> str:
    mean, std = summarize_scores(scores)
    return f"{mean:.4f} ± {std:.4f}"


def get_list(var: Any):
    return var if isinstance(var, list) else [var]


def file_name_from_pattern(pattern: str, lang: str, split: str):
    file_name = pattern.replace("LANG", lang)
    file_name = file_name.replace("SPLIT", split)
    return file_name


def beam_width_from_combination(combination: dict) -> Optional[str]:
    beam_width = combination.get("beam-width")
    if beam_width is None:
        return None
    if isinstance(beam_width, bool):
        raise ValueError(
            f"beam-width must be a nonnegative integer: {beam_width!r}.")
    try:
        beam_width_int = int(beam_width)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"beam-width must be a nonnegative integer: {beam_width!r}."
        ) from exc
    if beam_width_int != beam_width and str(beam_width_int) != str(beam_width):
        raise ValueError(
            f"beam-width must be a nonnegative integer: {beam_width!r}.")
    if beam_width_int < 0:
        raise ValueError(f"beam-width must be >= 0: {beam_width_int}.")
    if beam_width_int == 0:
        return None
    return f"beam{beam_width_int}"


def require_files(paths: List[str]) -> None:
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing expected experiment artifacts:\n" +
            "\n".join(missing)
        )


def validate_config(config_dict: dict) -> None:
    if config_dict["runs_per_model"] < 1:
        raise ValueError("runs_per_model must be >= 1.")
    seeds = config_dict.get("seeds")
    if seeds is not None:
        if (
                isinstance(seeds, (str, bytes, dict)) or
                not isinstance(seeds, Sequence)):
            raise ValueError("seeds must be a list of integer seeds.")
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("seeds must be a list of integer seeds.")
        if len(seeds) != config_dict["runs_per_model"]:
            raise ValueError("len(seeds) must equal runs_per_model.")
    languages = config_dict["data"]["languages"]
    if not languages:
        raise ValueError("data.languages must not be empty.")
    pattern = config_dict["data"]["pattern"]
    if "LANG" not in pattern or "SPLIT" not in pattern:
        raise ValueError("data.pattern must contain LANG and SPLIT placeholders.")

    configured_languages = set(languages)
    required_files = set()
    data_path = config_dict["data"]["path"]
    for grid_config in config_dict["grids"].values():
        for par in LANGUAGE_SPECIFIC_PARAMETERS:
            if par not in grid_config:
                continue
            if not isinstance(grid_config[par], dict):
                raise ValueError(f"{par} must map language names to file paths.")
            unknown_languages = set(grid_config[par]) - configured_languages
            if unknown_languages:
                raise ValueError(
                    f"{par} contains unknown languages: "
                    f"{sorted(unknown_languages)}")
        if seeds is not None and "pytorch-seed" in grid_config:
            raise ValueError(
                "Do not specify both top-level seeds and grid-level pytorch-seed.")
        for _, combination in grid_search_combinations(grid_config)[1].items():
            beam_width_from_combination(combination)

    for lang in languages:
        dev_file = file_name_from_pattern(pattern, lang, "dev")
        required_files.add(f"{data_path}/{dev_file}")
        for grid_config in config_dict["grids"].values():
            if not (
                    "precomputed-train" in grid_config and
                    lang in grid_config["precomputed-train"]):
                train_file = file_name_from_pattern(pattern, lang, "train")
                required_files.add(f"{data_path}/{train_file}")
            for par in LANGUAGE_SPECIFIC_PARAMETERS:
                if par in grid_config and lang in grid_config[par]:
                    required_files.add(grid_config[par][lang])
    require_files(sorted(required_files))


def build_option_args(config: dict) -> List[str]:
    parsed_args = []
    for par_name, par_value in config.items():
        if isinstance(par_value, bool):
            if par_value:
                parsed_args.append(f"--{par_name}")
            elif par_name in TRUE_DEFAULT_BOOLEAN_PARAMETERS:
                parsed_args.append(f"--no-{par_name}")
        elif isinstance(par_value, (list, tuple)):
            parsed_args.extend([f"--{par_name}", *[str(v) for v in par_value]])
        elif par_name in ['sed-params', 'precomputed-train', 'vocabulary']:
            continue
        else:
            parsed_args.extend([f"--{par_name}", str(par_value)])
    return parsed_args


def grid_search_combinations(grid_config: dict):
    search_config = {
        k: v for k, v in grid_config.items()
        if k not in LANGUAGE_SPECIFIC_PARAMETERS
    }
    nm_pairs = [[(k, v) for v in get_list(search_config[k])] for k in search_config]
    combinations = itertools.product(*nm_pairs)

    args_list, comb_dict = [], {}
    for i, c in enumerate(combinations, 1):
        args_dict = dict(c)
        args_list.append(build_option_args(args_dict))
        comb_dict[i] = args_dict
    return args_list, comb_dict


def build_train_command(extra_args: List[str]) -> List[str]:
    return ["trans-train", *extra_args]


def build_ensemble_command(gold: str, systems: List[str], output: str) -> List[str]:
    return [
        "trans-ensemble",
        "--gold", gold,
        "--systems", *systems,
        "--output", output,
    ]


def run_ensemble(gold: str, systems: List[str], output: str):
    process = subprocess.Popen(build_ensemble_command(gold, systems, output))
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Ensemble command failed with return code {return_code}: "
            f"{' '.join(build_ensemble_command(gold, systems, output))}")


def write_to_results_file(results_file: str, results: List[dict]):
    with open(results_file, "w") as f:
        for r in sorted(results, key=lambda x: x['dev_greedy_mean'], reverse=True):
            f.write(r['c_dir'] + "\n")
            f.write(f"dev\ngreedy: {r['dev_greedy']}\n")
            if r['dev_beam'] is not None:
                f.write(f"{r['beam_width']}: {r['dev_beam']}\n")
            if r['test_greedy'] is not None:
                f.write(f"test\ngreedy: {r['test_greedy']}\n")
            if r['test_beam'] is not None:
                f.write(f"{r['beam_width']}: {r['test_beam']}\n\n")
            else:
                f.write("\n")


def write_experiment_metadata(output_dir: str, config_dict: dict,
                              comb_dict: dict, parallel_jobs: int,
                              ensemble: bool) -> None:
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(f"{output_dir}/combinations.json", "w") as f:
        json.dump(comb_dict, f, indent=4)
    with open(f"{output_dir}/run_metadata.json", "w") as f:
        json.dump({
            "parallel_jobs": parallel_jobs,
            "ensemble": ensemble,
        }, f, indent=4)


def main(args: argparse.Namespace):
    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as config_file:
        config_dict = json.load(config_file)

    validate_config(config_dict)
    parallel_jobs = args.parallel_jobs
    if parallel_jobs is None:
        parallel_jobs = default_parallel_jobs(config_dict)
    validate_parallel_jobs(parallel_jobs)

    process_manager = ProcessManager(parallel_jobs)
    seeds = config_dict.get("seeds")
    try:
        for name, grid_config in config_dict["grids"].items():
            os.makedirs(f"{args.output}/{name}", exist_ok=True)

            # parse args
            args_list, comb_dict = grid_search_combinations(grid_config)
            write_experiment_metadata(
                f"{args.output}/{name}", config_dict, comb_dict,
                parallel_jobs, args.ensemble)

            # train
            for i, args_ in enumerate(args_list, 1):
                for lang in config_dict['data']['languages']:
                    for j in range(1, config_dict['runs_per_model']+1):
                        # reset ext_args
                        ext_args = args_.copy()
                        if seeds is not None:
                            ext_args.extend(["--pytorch-seed", str(seeds[j - 1])])

                        output = f"{args.output}/{name}/{lang}/{i}/{i}.{j}"

                        for par in ['sed-params', 'vocabulary']:
                            if par in grid_config and lang in grid_config[par]:
                                ext_args.extend(
                                    [
                                        "--"+par, grid_config[par][lang]
                                    ]
                                )

                        # create file names from pattern
                        dev_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'dev')
                        test_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'test')

                        dev = f"{config_dict['data']['path']}/{dev_file}"
                        test = f"{config_dict['data']['path']}/{test_file}"

                        # for train it's only needed if --train-precomputed is not specified
                        if not ('precomputed-train' in grid_config and lang in grid_config['precomputed-train']):
                            train_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'train')
                            train = f"{config_dict['data']['path']}/{train_file}"
                            train_par = ("--train", train)
                        else:
                            train_par = ("--precomputed-train", grid_config['precomputed-train'][lang])

                        ext_args.extend(
                            [
                                "--output", output,
                                *train_par,
                                "--dev", dev
                             ]
                        )

                        if os.path.exists(test):
                            ext_args.extend(
                                [
                                    "--test", test
                                ]
                            )

                        process_manager.wait_for_slot()
                        command = build_train_command(ext_args)
                        process_manager.start(
                            command,
                            output=output,
                            grid=name,
                            language=lang,
                            combination=i,
                            run=j,
                        )

        # all trainings in progress, stay in script so all processes can be aborted
        process_manager.wait_all()
    finally:
        process_manager.terminate_all()

    # evaluate: average of results per combination and ensemble
    for name, grid_config in config_dict["grids"].items():
        _, comb_dict = grid_search_combinations(grid_config)
        for lang in config_dict["data"]["languages"]:

            results = []  # average of single models
            ensemble_results = []  # ensemble results
            output_path = f"{args.output}/{name}/{lang}"
            dev_file =\
                f"{config_dict['data']['path']}/{file_name_from_pattern(config_dict['data']['pattern'], lang, 'dev')}"
            test_file =\
                f"{config_dict['data']['path']}/{file_name_from_pattern(config_dict['data']['pattern'], lang, 'test')}"
            has_test = os.path.exists(test_file)

            # level: combination
            for c_dir in [str(i) for i in sorted(comb_dict)]:  # c_dir == name of combination (number)
                dev_beam_scores, dev_greedy_scores = [], []
                test_beam_scores, test_greedy_scores = [], []
                c_dir_path = f"{output_path}/{c_dir}"  # directory of combination
                combination = comb_dict[int(c_dir)]
                run_dirs = [
                    f"{c_dir}.{i}"
                    for i in range(1, config_dict["runs_per_model"] + 1)
                ]
                n_runs = len(run_dirs)

                beam_width = beam_width_from_combination(combination)
                expected_files = [
                    f"{c_dir_path}/{c_run}/dev_greedy.eval"
                    for c_run in run_dirs
                ]
                if beam_width:
                    expected_files.extend(
                        f"{c_dir_path}/{c_run}/dev_{beam_width}.eval"
                        for c_run in run_dirs
                    )
                if has_test:
                    expected_files.extend(
                        f"{c_dir_path}/{c_run}/test_greedy.eval"
                        for c_run in run_dirs
                    )
                    if beam_width:
                        expected_files.extend(
                            f"{c_dir_path}/{c_run}/test_{beam_width}.eval"
                            for c_run in run_dirs
                        )
                if args.ensemble:
                    for split in ["dev", "test"] if has_test else ["dev"]:
                        expected_files.extend(
                            f"{c_dir_path}/{c_run}/{split}_greedy.predictions"
                            for c_run in run_dirs
                        )
                        if beam_width:
                            expected_files.extend(
                                f"{c_dir_path}/{c_run}/{split}_{beam_width}.predictions"
                                for c_run in run_dirs
                            )
                require_files(expected_files)

                # level: run per combination
                for c_run in run_dirs:
                    # dev greedy
                    dev_greedy_scores.append(last_value_from_file(
                        f"{c_dir_path}/{c_run}/dev_greedy.eval"))

                    # dev beam
                    if beam_width:
                        dev_beam_scores.append(last_value_from_file(
                            f"{c_dir_path}/{c_run}/dev_{beam_width}.eval"))

                    if has_test:
                        # test greedy
                        test_greedy_scores.append(last_value_from_file(
                            f"{c_dir_path}/{c_run}/test_greedy.eval"))
                        # test beam
                        if beam_width:
                            test_beam_scores.append(last_value_from_file(
                                f"{c_dir_path}/{c_run}/test_{beam_width}.eval"))

                result = {
                    'c_dir': c_dir,
                    'beam_width': beam_width,
                    'dev_greedy_mean': summarize_scores(dev_greedy_scores)[0],
                    'dev_greedy': format_score_summary(dev_greedy_scores),
                    'dev_beam': format_score_summary(dev_beam_scores)
                    if beam_width else None,
                    'test_greedy': format_score_summary(test_greedy_scores)
                    if has_test else None,
                    'test_beam': format_score_summary(test_beam_scores)
                    if has_test and beam_width else None
                }
                results.append(result)

                if args.ensemble:
                    result = {
                        'c_dir': c_dir,
                        'beam_width': beam_width,
                        'dev_greedy_mean': 0.,
                        'dev_beam': None,
                        'test_greedy': None,
                        'test_beam': None
                    }
                    golds = [('dev', dev_file), ('test', test_file)] if has_test else [('dev', dev_file)]
                    for split, gold_file in golds:
                        systems =\
                            [f"{c_dir_path}/{c_run}/{split}_greedy.predictions" for c_run in run_dirs]
                        # greedy
                        run_ensemble(gold_file, systems, f"{c_dir_path}/greedy_ensemble")
                        result[f"{split}_greedy"] =\
                            round(last_value_from_file(f"{c_dir_path}/greedy_ensemble/{split}_{n_runs}ensemble.eval"), 4)
                        if split == "dev":
                            result["dev_greedy_mean"] = result[f"{split}_greedy"]
                        # beam
                        if beam_width:
                            systems = \
                                [f"{c_dir_path}/{c_run}/{split}_{beam_width}.predictions" for c_run in run_dirs]
                            run_ensemble(gold_file, systems, f"{c_dir_path}/{beam_width}_ensemble")
                            result[f"{split}_beam"] = \
                                round(last_value_from_file(f"{c_dir_path}/{beam_width}_ensemble/{split}_{n_runs}ensemble.eval"), 4)
                    ensemble_results.append(result)

            # write to results text file
            write_to_results_file(f"{args.output}/{name}/{lang}/results.txt", results)

            if args.ensemble:
                write_to_results_file(f"{args.output}/{name}/{lang}/ensemble_results.txt", ensemble_results)


def cli_main():
    parser = argparse.ArgumentParser(
        description="Grid search.")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument("--parallel-jobs", type=int,
                        help="Max number of parallel trainings. Defaults to 30 for CPU-only grids and 4 if any grid uses a non-CPU device.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Produce ensemble results.")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

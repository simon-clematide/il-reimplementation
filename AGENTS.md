# Project Notes for Agents

## Project Summary

This repository contains a Python package named `neural_transducer` with the
import package `trans`. It implements a CLI-based neural string transducer,
originally for SIGMORPHON-style string transduction tasks and now also used in
the `notebooks/g2p.ipynb` workflow for grapheme-to-phoneme (G2P) training and
evaluation.

The core model is based on Makarov and Clematide (2020), with later support for
GPU mini-batch training, batched greedy decoding, beam search, LSTM encoders,
and transformer encoders.

## Repository Layout

- `trans/train.py`: main `trans-train` CLI. Loads TSV data, builds or loads
  vocabularies, trains or loads SED alignment parameters, precomputes expert
  rollouts, trains the neural transducer, and writes dev/test predictions.
- `trans/transducer.py`: PyTorch model, decoder logic, greedy decoding, beam
  search decoding, valid-action masking, and loss computation.
- `trans/sed.py`: stochastic edit distance model used for alignment/expert
  scoring.
- `trans/optimal_expert.py` and `trans/optimal_expert_substitutions.py`: expert
  policies for rollouts during imitation-style training.
- `trans/vocabulary.py`: character, action, and feature vocabularies.
- `trans/actions.py`: edit action classes.
- `trans/encoders.py`: LSTM and transformer encoder implementations.
- `trans/ensembling.py`: `trans-ensemble` majority-vote prediction ensembling.
- `trans/grid_search.py`: `trans-grid-search` orchestration for multiple
  training runs and optional ensembling.
- `trans/tests/`: unit tests for the lower-level transducer, expert, SED,
  vocabulary, and utility behavior.
- `trans/test_data/`: small SIGMORPHON-derived fixture data.
- `trans/docs/`: grid-search config schema and example.
- `notebooks/g2p.ipynb`: Colab-oriented application notebook for G2P training
  on SIGMORPHON 2020/2021 datasets.

## Data Format

The training workflow expects UTF-8 TSV files:

- Without features: `input<TAB>target`
- With UniMorph-style features: `input<TAB>target<TAB>feature1;feature2;...`
- Test data can omit the target in the non-feature case.

The notebook uses SIGMORPHON 2020/2021 task data and demonstrates Italian
low-resource G2P training.

## Commands

When running local project commands yourself in this environment, use `remake`
instead of `make`.

Examples:

- `remake test`
- `remake -n world-test TEST_INPUT_DIR=data/test`
- `remake run-baseline RUN_BASELINE_ARGS='--max-docs 1'`

When editing user-facing documentation, release notes, README examples,
Makefile help text, or shell snippets, write commands as `make`, not `remake`.
Do not mention `remake` in public-facing documentation unless explicitly asked.

Current project metadata also supports direct Python/CLI usage:

- `trans-train --help`
- `trans-ensemble --help`
- `trans-grid-search --help`
- `python setup.py test` as documented in the README

## Dependency Notes

There is a mismatch between `setup.py`, `pyproject.toml`, and
`requirements.txt`:

- `setup.py` lists the historical runtime dependencies, including `torch`,
  `editdistance`, `numpy`, `progressbar`, and `scipy`.
- `pyproject.toml` currently lists only `Cython` as an active dependency; the
  other runtime dependencies are commented out.
- `requirements.txt` also comments out some important runtime dependencies.

This matters for the Colab notebook because installing from Git via pip will
use the PEP 517 project metadata and may not install everything needed by
`trans-train`.

## Review Context

Recent review targets identified in this codebase:

- `trans/train.py`: gradient accumulation currently steps on batch index `0`
  and can miss the trailing accumulated gradients.
- `trans/grid_search.py`: ensemble subprocess command construction is missing a
  comma after `"trans-ensemble"`.
- `trans/utils.py`: `Dataset.to()` does not move `encoded_features`, which can
  break feature-based precomputed training on CUDA.
- `trans/transducer.py`: `valid_actions_lookup` is built for a fixed range of
  `MAX_INPUT_SEQ_LEN = 100`, but decoding can index one past that for
  99-character inputs.
- `notebooks/g2p.ipynb`: the notebook is Colab/GPU-oriented and assumes
  `--device cuda`; add CPU fallback or a runtime check before making it a robust
  teaching/application notebook.


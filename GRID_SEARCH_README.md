# Grid Search

`trans-grid-search` runs repeated `trans-train` jobs for every combination in a
JSON hyperparameter grid, then aggregates the generated evaluation files into
ranked result tables. It is intended for controlled research sweeps where each
combination is trained multiple times and optionally ensembled.

## Command

```bash
trans-grid-search --config grid.json --output runs/grid-search
```

Useful options:

```bash
trans-grid-search --config grid.json --output runs/grid-search --parallel-jobs 8
trans-grid-search --config grid.json --output runs/grid-search --ensemble
```

`--parallel-jobs` is optional. If omitted, the default is `30` for CPU-only
grids and `4` when any grid config contains a non-CPU `device` such as `mps` or
`cuda`. Pass the option explicitly for production runs so the experiment records
the intended concurrency.

## Configuration Model

The configuration has three top-level sections:

```json
{
  "data": {
    "path": "data.d/sigmorphon",
    "pattern": "LANG_SPLIT.tsv",
    "languages": ["ita", "deu"]
  },
  "runs_per_model": 3,
  "grids": {
    "lstm": {
      "enc-type": "lstm",
      "batch-size": [32, 64],
      "beam-width": [0, 5]
    }
  }
}
```

`data.path` is the directory containing TSV files. `data.pattern` must contain
both `LANG` and `SPLIT`; these placeholders are replaced for each language and
split. With the example above, Italian files are expected at:

```text
data.d/sigmorphon/ita_train.tsv
data.d/sigmorphon/ita_dev.tsv
data.d/sigmorphon/ita_test.tsv
```

`train` and `dev` are required unless a language has `precomputed-train`.
`test` is optional. If a test file exists, every successful run is expected to
produce test evaluation artifacts.

`runs_per_model` controls replicate count per language and combination. The
runner creates directories such as `1.1`, `1.2`, `1.3` for combination `1`.

Each entry under `grids` is a named sweep. Scalar values are passed unchanged to
`trans-train`; list values create Cartesian-product dimensions. Booleans are
treated as CLI flags: `true` emits `--flag`, `false` emits nothing.

## Language-Specific Files

These keys are not hyperparameter dimensions:

```text
sed-params
precomputed-train
vocabulary
```

They are language-specific mappings and are excluded from the Cartesian product.

```json
{
  "grids": {
    "with_sed": {
      "beam-width": [0, 5],
      "sed-params": {
        "ita": "data.d/sed/ita.pkl",
        "deu": "data.d/sed/deu.pkl"
      },
      "vocabulary": {
        "ita": "data.d/vocab/ita.pkl"
      }
    }
  }
}
```

Unknown language keys are rejected during preflight validation, so a typo such
as `"tia"` fails before jobs are launched.

## Example: CPU Sweep

```json
{
  "data": {
    "path": "data.d/sigmorphon2021/low",
    "pattern": "LANG_SPLIT.tsv",
    "languages": ["ita"]
  },
  "runs_per_model": 3,
  "grids": {
    "lstm_small": {
      "device": "cpu",
      "pytorch-seed": [1, 2, 3],
      "enc-type": "lstm",
      "char-dim": 64,
      "action-dim": 64,
      "enc-hidden-dim": [64, 128],
      "dec-hidden-dim": [64, 128],
      "batch-size": 32,
      "epochs": 100,
      "patience": 20,
      "beam-width": [0, 5],
      "sed-em-mode": "damped",
      "sed-em-damping": 0.9
    }
  }
}
```

Run:

```bash
trans-grid-search --config grid_cpu.json --output runs/ita-cpu --parallel-jobs 30
```

## Example: MPS/GPU Sweep

```json
{
  "data": {
    "path": "data.d/sigmorphon2021/low",
    "pattern": "LANG_SPLIT.tsv",
    "languages": ["ita"]
  },
  "runs_per_model": 2,
  "grids": {
    "mps_lstm": {
      "device": "mps",
      "pytorch-seed": [1, 2],
      "enc-type": "lstm",
      "char-dim": 128,
      "action-dim": 128,
      "enc-hidden-dim": 128,
      "dec-hidden-dim": 128,
      "batch-size": 64,
      "eval-batch-size": 64,
      "epochs": 100,
      "patience": 20,
      "beam-width": 5
    }
  }
}
```

Run:

```bash
trans-grid-search --config grid_mps.json --output runs/ita-mps
```

Because the grid uses `device: "mps"`, omitted `--parallel-jobs` defaults to
`4`. Override it if your machine needs lower or higher concurrency.

## Output Layout

For a grid named `lstm`, language `ita`, combination `1`, and three runs:

```text
runs/grid-search/
  lstm/
    combinations.json
    ita/
      1/
        1.1/
          best.model
          dev_greedy.eval
          dev_greedy.predictions
          dev_beam5.eval
          dev_beam5.predictions
        1.2/
        1.3/
      results.txt
      ensemble_results.txt
```

`combinations.json` maps combination ids to hyperparameters. Evaluation uses
the expected run names from `runs_per_model`; ensemble directories are ignored
when averaging.

`results.txt` contains average scores across runs, sorted by `dev_greedy`.
Beam labels are stored per result row, so combinations with different beam
widths are reported correctly.

## Ensembles

With `--ensemble`, the runner calls `trans-ensemble` for each combination using
the per-run prediction files. The expected prediction files are validated before
ensemble execution. Ensemble outputs are written into directories such as:

```text
greedy_ensemble/
beam5_ensemble/
```

and summarized in `ensemble_results.txt`.

## Validation And Failure Behavior

Before launching training jobs, the runner validates:

- `runs_per_model >= 1`
- nonempty `data.languages`
- `data.pattern` contains `LANG` and `SPLIT`
- required train/dev files exist
- language-specific `sed-params`, `precomputed-train`, and `vocabulary` paths exist
- language-specific mappings contain only configured languages
- `beam-width` is a nonnegative integer

After training, expected evaluation and prediction artifacts are validated
before averaging or ensembling. Missing artifacts fail with a contextual
`FileNotFoundError`; malformed evaluation files fail with a parse error that
includes the path.

Training subprocess failures report grid name, language, combination, run,
output directory, return code, and command.

## Reproducibility Notes

The grid runner currently passes through whatever seed configuration you put in
the grid, most commonly `pytorch-seed`. If you want deterministic replicates,
make seed values explicit in the grid, for example:

```json
{
  "pytorch-seed": [1, 2, 3]
}
```

Be aware that `runs_per_model` and a seed list are independent Cartesian
dimensions. If you set both `runs_per_model: 3` and `pytorch-seed: [1, 2, 3]`,
you will run nine jobs per non-seed combination. For a one-to-one mapping from
replicate number to seed, the runner would need an explicit seed-list feature;
that is not implemented yet.

## Current Limitations

There is no resume or overwrite mode. Existing output directories are reused
and `combinations.json` is overwritten. For professional experiment tracking,
prefer a fresh output directory per sweep until explicit `--resume` and
`--overwrite` semantics are added.

Selection is currently implicit: result files are sorted by `dev_greedy`. If
your protocol selects hyperparameters by beam performance, document that
outside this runner or add a selection-metric option before comparing sweeps.

The scheduler limits only the number of subprocesses. It does not implement
separate CPU/GPU resource pools, so mixed CPU and accelerator grids use one
global concurrency limit.

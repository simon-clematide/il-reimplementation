# A neural transducer
Shortcuts: [Introduction](https://github.com/slvnwhrl/il-reimplementation#introduction) -
[SIGMORPHON2022 shared task models](https://github.com/slvnwhrl/il-reimplementation#sigmorphon2022-shared-task-models) -
[Installation](https://github.com/slvnwhrl/il-reimplementation#installation) -
[Usage](https://github.com/slvnwhrl/il-reimplementation#usage) -
[Citation](https://github.com/slvnwhrl/il-reimplementation#citation) - 
[References](https://github.com/slvnwhrl/il-reimplementation#references)

## Introduction
This package contains a cli-based neural transducer for string transduction tasks. It was successfully used in the 
SIGMORPHON 2022 shared task on [morpheme segmentation](https://github.com/sigmorphon/2022SegmentationST) and
[morphological inflection](https://github.com/sigmorphon/2022InflectionST)!
<br><br>
The transducer builds on the model by [Makarov & Clematide (2020)](https://aclanthology.org/2020.sigmorphon-1.19).
This implementation introduces GPU-supported mini-batch training and batched greedy decoding as well as support for
transformer-based encoders. See [Wehrli & et al. (2022)](https://aclanthology.org/2022.sigmorphon-1.21) for more infos. :)

## SIGMORPHON2022 shared task models
We received some requests to share our models from our successful submission to the shared task on
[morpheme segmentation](https://github.com/sigmorphon/2022SegmentationST) and we are happy to share the models!
So if you want to use these models with this package, have a look at [this repository](https://github.com/slvnwhrl/sigmorphon2022-models) where we host the models.

## Installation
Please make sure that you are using Python 3.10 or newer, up to Python 3.13.
To install this package, perform the following steps:

* Clone the repository and change to the package directory:

        git clone https://github.com/slvnwhrl/il-reimplementation.git neural_transducer
        cd neural_transducer

* Create and activate a supported Python virtual environment:

        python3 -m venv .venv
        source .venv/bin/activate

* Install the package and its runtime dependencies:

  * default installation

        pip install .

  * local development (without the need to reinstall the package after changes):

        pip install -e .

PyTorch is installed from the package metadata. For a CUDA-specific PyTorch build,
install the appropriate PyTorch wheel for your platform first, following the
official PyTorch selector, and then install this package.

* Optionally, run unit tests:

        python -m unittest discover -s trans/tests -v

## Usage
### Training
In order to train a model, directly run the python script ``train.py`` 
via ``python train.py`` or use the cli entry point ``trans-train``.

The most important (and required) parameters are:
* ``--train`` path to the training data
* ``--dev`` path to the development data
* ``--output`` path to the output directory

For a full list of available training configurations, use ``trans-train --help``.

By default, source and target strings are treated as character sequences. For
phone-like outputs where symbols are separated by spaces, set a target
separator:

        --source-separator none --target-separator " "

For example, the target field ``a d͡ʒ e n t͡s i a`` is then treated internally as
the token sequence ``["a", "d͡ʒ", "e", "n", "t͡s", "i", "a"]`` rather than as
individual Unicode characters and spaces. The separators are stored in
``vocabulary.pkl`` and checkpoint metadata.

For LSTM encoders, ``--enc-dropout`` is the legacy PyTorch inter-layer LSTM
dropout. It only has an effect when ``--enc-layers`` is greater than 1. For
explicit dropout on the encoder output sequence, including one-layer BiLSTMs,
use ``--enc-output-dropout``. The default output-dropout type is locked dropout:

        --enc-output-dropout 0.3 --enc-output-dropout-type locked

The decoder can optionally receive feedback about the symbol emitted by the
previous action. This is disabled by default and preserves the historical
action-feedback architecture:

        --output-feedback-dim 0

Set a positive dimension, for example ``--output-feedback-dim 32``, to append a
previous-output embedding to the decoder input. The feedback symbols include
``<BOS_OUTPUT>``, ``<NO_OUTPUT>``, and the target symbols used by insertion and
substitution actions.

The default training objective is the hard set-valued oracle loss. To give
near-optimal expert actions partial credit according to their SED cost gap, set
a positive soft-oracle temperature:

        --expert-temperature 2

At temperature ``0.0`` the historical hard oracle loss is used.
For positive temperatures, ``--expert-loss marginal`` keeps the weighted
oracle-mass objective, while ``--expert-loss normalized_ce`` trains against the
normalized soft expert distribution over finite expert-supported actions:

        --expert-temperature 2 --expert-loss normalized_ce

The fixed-margin expert loss ranks at least one expert-optimal action above all
other decoder-valid actions by a logit margin. It does not use the SED cost-gap
magnitudes or ``--expert-temperature``:

        --expert-loss margin --expert-margin 1.0

SED parameters are estimated during training unless ``--sed-params`` points to
an existing ``sed.pkl`` file. The default SED estimator is damped EM:

        --sed-em-mode damped --sed-em-damping 0.9

Use ``--sed-em-mode strict`` for the paper-faithful Ristad-Yianilos EM update.
When training fits a new SED model, it writes both ``sed.pkl`` and
``sed.pkl.json``. The JSON sidecar records the training input path, token
separators, EM settings, corpus size, alphabets, git commit, and full training
arguments.

#### Training option compatibility

The command-line help shows all defaults:

        trans-train --help

Some options are alternatives or only meaningful in specific modes:

* ``--train`` is required for ordinary training. If ``--precomputed-train`` is
  used instead, ``--vocabulary`` must also be supplied. Conversely,
  ``--precomputed-train`` and ``--vocabulary`` are a pair: do not provide only
  one of them.
* ``--save-precomputed-train`` only has an effect when training examples are
  precomputed from ``--train`` during the current run.
* ``--source-separator`` and ``--target-separator`` define the tokenization used
  to build the vocabulary, SED expert, action vocabulary, checkpoints, and
  diagnostics. Use the same separator settings when creating ``sed.pkl``,
  training the neural model, and running diagnostics. If you load an existing
  ``--vocabulary`` or model metadata, its stored separators should be treated as
  authoritative.
* ``--sed-params`` loads an existing SED model. If it is omitted, training fits a
  new SED model using ``--sed-em-iterations``, ``--sed-em-mode``, and
  ``--sed-em-damping``. These EM options do not refit or modify an already
  supplied ``--sed-params`` file.
* ``--sed-em-mode strict`` is the paper-faithful EM estimator.
  ``--sed-em-mode damped`` is the project default for backward-compatible
  training behavior; it interpolates the strict EM estimate with the previous
  parameters using ``--sed-em-damping``.
* ``--expert-temperature 0`` uses the historical hard set-valued oracle loss.
  With positive temperatures, ``--expert-loss marginal`` and
  ``--expert-loss normalized_ce`` use SED cost gaps and therefore require expert
  action-cost precomputation.
* ``--expert-loss margin`` ignores ``--expert-temperature`` and SED cost-gap
  magnitudes. It uses ``--expert-margin`` to rank the best expert-optimal action
  above decoder-valid alternatives.
* If you use ``--precomputed-train`` from an older run, regenerate it after
  changing options that alter expert/training tensors, especially
  ``--source-separator``, ``--target-separator``, ``--output-feedback-dim``, or
  positive-temperature expert losses. Periodic model roll-in refreshes
  regenerate trajectories from each sample's stored input and target, so loaded
  precomputed data must come from a compatible tokenizer/vocabulary setup.
* ``--output-feedback-dim 0`` disables previous-output feedback. A positive
  value requires output histories in precomputed training data; newly
  precomputed data includes them automatically.
* ``--rollin-prob 0`` disables imitation-learning trajectory refreshes and
  reproduces fixed expert trajectories. With a positive value, training starts
  from expert trajectories and periodically refreshes cached trajectories from
  states partly induced by the current model. ``--rollin-start`` selects the
  first refresh epoch, ``--rollin-refresh`` selects the refresh period, and
  ``--rollin-policy greedy`` uses the current greedy decoder-valid model action
  when roll-in is selected. ``--rollin-seed`` controls only the model/expert
  roll-in choices and defaults to ``--pytorch-seed`` when available. Expert
  supervision is still computed by the SED expert at every visited state;
  roll-in only changes the visited state distribution. Roll-in trajectories are
  capped by the smaller of the global decoder limit and a source/target-length
  dependent bound, and refresh logs report truncations.
* For LSTM encoders, ``--enc-dropout`` affects only stacked recurrent encoders
  with ``--enc-layers`` greater than 1. Use ``--enc-output-dropout`` for
  one-layer encoders. ``--enc-output-dropout-type none`` disables explicit output
  dropout regardless of the numeric dropout value.
* Scheduler-specific options such as ``--factor``, ``--lrs-patience``, and
  ``--cooldown`` are only used with ``--scheduler reduce_on_plateau``.
  Optimizer-specific options are similarly used only by the selected
  ``--optimizer``.
* ``--eval-batch-size`` defaults to ``--batch-size`` when omitted. Beam search
  evaluation is enabled only when ``--beam-width`` is greater than 0; greedy
  decoding is always evaluated. Beam search keeps the live beam at the requested
  width and returns up to that many completed hypotheses per input. With
  ``--verbose``, beam decoding logs debug counters such as active beam size,
  decoder calls, expansions, steps, and completed hypotheses. As a temporary
  workaround for slow accelerator beam decoding, final beam evaluation runs from
  a CPU copy of the best checkpoint when training uses ``--device mps`` or
  ``--device cuda``; greedy evaluation remains on the selected device.
* Development-set checkpoint selection maximizes exact string accuracy first.
  If two epochs have the same string accuracy, the tie-breaker is higher symbol
  accuracy, computed as ``1 -`` total token-level Levenshtein distance divided
  by the total number of reference symbols. String accuracy and symbol accuracy
  are reported in logs, checkpoint metadata, and evaluation files.
* ``--device mps``/``cuda`` runs the neural model on the accelerator. Expert
  precomputation and mask construction are CPU-side bookkeeping and are moved to
  the selected device before neural training.

Common controlled objective comparisons:

        # Historical hard oracle
        --expert-temperature 0 --expert-loss marginal

        # Cost-sensitive marginal oracle mass
        --expert-temperature 2 --expert-loss marginal

        # Normalized soft expert distribution
        --expert-temperature 2 --expert-loss normalized_ce

        # Fixed-margin logit ranking; temperature is unused
        --expert-loss margin --expert-margin 1.0

        # Periodic imitation-learning trajectory refresh
        --rollin-prob 0.2 --rollin-start 5 --rollin-refresh 5 --rollin-seed 42

### Ensembling
To ensemble a number of models based on majority voting, run the python script 
``ensembling.py`` via ``python ensembling.py`` or use the cli entry point 
``trans-ensemble``. The following parameters are required:
* ``--gold`` path to the gold data
* ``--systems`` path to the systems' data
* ``--output`` path to the output directory

### Grid Search
In order to enable efficient model (hyper)parameter exploration,
this package offers grid search that allows to run a defined number of models
for specified configurations. To specify configurations, 
a JSON file is used (see below for further explanations).
To run grid search, run the python script ``grid_search.py`` via 
``python grid_search.py`` or use the cli entry point ``trans-grid-search``. 

The following parameters are available:
* ``--config`` path to the JSON config file (required)
* ``--output`` path to the output directory (required)
* ``--parallel-jobs`` number of jobs (i.e., trainings) that are run in parallel (on CPU and GPU)
* ``--ensemble`` bool indicating whether to produce ensemble results or not

The command ``trans-grid-search --help`` can be run to get information about 
the available parameters.

### SED Analysis
To inspect source-target pairs that are surprising under a fitted stochastic
edit-distance model, use ``trans-analyze-sed``. The following parameters are
required:
* ``--sed-params`` path to a fitted ``sed.pkl`` file
* ``--input`` path to a TSV file with source and target in the first two columns

Example:

        trans-analyze-sed --sed-params data.d/sed-2021/low_ita/sed.pkl \
          --input data.d/sigmorphon2021/low/ita_train.tsv \
          --output data.d/ita_train_sed_analysis.tsv \
          --sort-by target_length_surprisal

The output is a TSV table with stochastic surprisal, normalized surprisal,
Viterbi surprisal, alignment ambiguity, and the best Viterbi alignment.

### Decoder Diagnostics
To inspect incorrect greedy predictions against the SED-backed expert policy,
use ``trans-diagnose`` after training:

        trans-diagnose \
          --model data.d/result/best.model \
          --metadata data.d/result/best.model.json \
          --vocabulary data.d/result/vocabulary.pkl \
          --sed-params data.d/result/sed.pkl \
          --input data.d/sigmorphon2021/low/ita_test.tsv \
          --output data.d/result/diagnostics \
          --device cpu

The command writes two TSV files:

* ``diagnostics.tsv``: one row per decoded example with summary statistics such
  as ``first_non_optimal_step``, ``num_non_optimal``, ``fraction_oracle_optimal``,
  and the oracle probability mass at the first policy deviation.
* ``diagnostic_steps.tsv``: one row per decoded action with the model action,
  model probability, expert-optimal action set, oracle-set probability mass,
  oracle margin, expert cost gaps, and top model actions.

#### Configuration file
The JSON-based configuration file needs to be passed via ``--config`` parameter.
It basically contains information about the used data as well as model (hyper)parameters.
An [example](trans/docs/grid_search_config_example.json) can be found in the docs folder. The schema for the JSON file is
defined as following:

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Grid search config",
  "description": "Configuration file for grid-based search/optimization",
  "type": "object",
  "properties": {
    "data": {
      "description": "Information about the used data",
      "type": "object",
      "properties": {
        "path": {
          "description": "The path to the training data",
          "type": "string"
        },
        "pattern": {
          "description": "The pattern used to find the split- and language-specific data files. In the given path the words SPLIT and LANG will be replaced.",
          "type": "string"
        },
        "languages": {
          "description": "List of languages to train on",
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      }
    },
    "required": [
      "path",
      "pattern",
      "languages"
    ]
  },
  "runs_per_model": {
    "description": "The number of models that are trained per possible combination of config parameters.",
    "type": "integer"
  },
  "grids": {
    "description": "This objects contains specification for named grids.",
    "type": "object",
    "patternProperties": {
      "^.*$": {
        "description": "This object represents a single grid and contains the model (hyper)parameters as key-value pairs. Note that values can either be passed as single values or as array of different values. An exception is the sed-params parameter which expects a object containing key-value pairs of languages and paths.",
        "type": "object"
      }
    }
  },
  "required": [
    "data",
    "runs_per_model",
    "grids"
  ]
}
```

In principle, all parameter values can either be passed as a single value or 
as an array of values. In any case, all possible combinations of all passed
parameter values for a specific grid will be produced and used for training. However,
two things should be noted:
* Firstly, for parameters without required values (e.g., ``--nfd``) a boolean needs
to be specified.
* Secondly, the model parameter ``--sed-params`` expects a dictionary that contains
key value pairs of language name and path to the sed parameters. If a key for a
specific language is missing, a new sed aligner will be trained.

#### Output structure
All output will be generated in the folder specified by the ``--output`` cli argument.
This folder contains a separate folder for each grid that is specified in the config folder.
The name is defined by the name used as key values in the ``grids`` property 
of the config file. This folder contains a `combinations.json` file that 
describes the different possible combinations and maps each combination to a number.
Additionally, this folder contains a separate folder for each trained language.
Each of these "language folders" contains a folder for each possible grid combination
(--> number from `combinations.json`) which, in turn, contain all the trained
models for this specific configuration. Additionally, a results text file is produced
that documents the performance average (accuracy) of all runs. If the ``--ensemble``
parameter is passed, separate results text files will be produced.

## Citation
If you use this package, please cite the following paper:
```
@inproceedings{wehrli-etal-2022-cluzh,
    title = "{CLUZH} at {SIGMORPHON} 2022 Shared Tasks on Morpheme Segmentation and Inflection Generation",
    author = "Wehrli, Silvan  and
      Clematide, Simon  and
      Makarov, Peter",
    booktitle = "Proceedings of the 19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sigmorphon-1.21",
    doi = "10.18653/v1/2022.sigmorphon-1.21",
}
```

## References
P. Makarov and S. Clematide. [CLUZH at SIGMORPHON 2020 Shared Task on Multilingual Grapheme-to-Phoneme Conversion](https://aclanthology.org/2020.sigmorphon-1.19). In *Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*, 2020.

S. Wehrli, S. Clematide, and P. Makarov. [CLUZH at SIGMORPHON 2022 Shared Tasks on Morpheme Segmentation and Inflection Generation](https://aclanthology.org/2022.sigmorphon-1.21). In *19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*, 2022.

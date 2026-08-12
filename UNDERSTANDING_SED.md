# Understanding SED in This Project

## What SED is

SED stands for stochastic edit distance. In this project it is a probabilistic
edit model over source and target strings. It assigns log probabilities to four
operation families:

- deletion: consume one source symbol and emit nothing
- insertion: emit one target symbol without consuming source
- substitution: consume one source symbol and emit one target symbol
- end of sequence: stop

For G2P, the source is usually a grapheme string and the target is a phone
string. A transformation such as:

```text
ciao -> t͡ʃ a o
```

can be explained by many different edit paths. SED gives a learned cost to
those paths instead of requiring a fixed hand-written alignment.

In code, SED is implemented in [trans/sed.py](/Users/siclemat/pj/2023/il-reimplementation/trans/sed.py). Its main class is
`StochasticEditDistance`.

## Why SED is used

The neural transducer predicts edit actions, but training needs to know which
actions are good at each decoder step. That is difficult because target strings
are not aligned to source strings.

SED solves this training-time alignment problem. It acts as a learned edit-cost
model. The expert uses those costs to decide which next action leaves the best
remaining path from the current partial prediction to the gold target.

At inference time, SED is not used. The trained neural transducer predicts
actions by itself.

```text
training:
  source,target -> SED/expert -> optimal action masks -> neural training

inference:
  source -> neural transducer -> predicted target
```

## The learned parameters

`StochasticEditDistance` stores parameters in a `ParamDict`:

```python
delta_sub[(source_symbol, target_symbol)]
delta_del[source_symbol]
delta_ins[target_symbol]
delta_eos
```

All values are log probabilities. Higher means more likely; lower means more
costly.

For example, after fitting Italian G2P data, SED may learn that:

- substituting `a -> a` is cheap
- substituting `c -> k` is plausible
- inserting a phone after some letters is sometimes needed
- deleting a source character is possible but usually less likely

The exact values come from data, not from linguistic rules.

## How SED is fitted

The model is fitted with expectation-maximization in `StochasticEditDistance.em()`.
The training data provides source/target pairs but not edit paths. EM estimates
which edit operations are likely to have generated the pairs.

Each EM iteration does:

1. `forward_evaluate(source, target)`: compute total probability of all edit
   paths from source to target.
2. `backward_evaluate(source, target)`: compute suffix probabilities.
3. `e_step(...)`: accumulate expected soft counts for insertions, deletions,
   substitutions, and EOS.
4. `m_step(...)`: normalize those counts into log probabilities.

The command used in the notebook-style workflow is:

```bash
.venv/bin/python -m trans.train \
  --train data.d/sigmorphon2021/low/ita_train.tsv \
  --dev data.d/sigmorphon2021/low/ita_dev.tsv \
  --sed-em-iterations 5 \
  --output /tmp/sed-ita-2021/low_ita \
  --epochs 1 \
  --beam-width 0 \
  --batch-size 200 \
  --device mps
```

The useful artifact is:

```text
/tmp/sed-ita-2021/low_ita/sed.pkl
```

which is copied to:

```text
data.d/sed-2021/low_ita/sed.pkl
```

## Dynamic programming views

SED has two related scoring modes.

`stochastic_distance(source, target)` sums over all possible edit paths. This
is useful for likelihood and EM.

```text
score = log sum over every valid edit sequence
```

`viterbi_distance(source, target)` finds the single best edit path.

```text
score = log probability of the best edit sequence
```

The implementation stores explicit Viterbi backpointers while filling the
dynamic-programming table. This is important: traceback must follow the
predecessor plus transition that produced a cell, not merely the predecessor
cell with the highest standalone score.

The final EOS probability is kept separate:

```text
alpha[T,V] = best edit-operation sequence before EOS
score      = alpha[T,V] + log P(EOS)
```

With alignment enabled, Viterbi can return the best edit sequence. For example:

```python
best_edits, score = sed.viterbi_distance(
    source="affa",
    target="iffig",
    with_alignment=True,
)
```

One plausible alignment is:

```text
a -> i
f -> f
f -> f
insert i
a -> g
```

That corresponds to these edit actions:

```text
Sub(a, i), Sub(f, f), Sub(f, f), Ins(i), Sub(a, g)
```

Under ties or after parameter changes, the exact alignment may differ. A stronger
invariant is that the returned alignment's log probability must equal the
reported Viterbi score.

```python
alignment, score = sed.viterbi_distance(source, target, with_alignment=True)
assert np.isclose(sed.alignment_log_probability(alignment), score)
```

## How the expert uses SED

The neural model does not call SED directly during its forward pass. Instead,
the training script creates an expert:

```python
sed_aligner = StochasticEditDistance.from_pickle(args.sed_params)
expert = OptimalSubstitutionExpert(sed_aligner)
```

The expert is implemented in [trans/optimal_expert_substitutions.py](/Users/siclemat/pj/2023/il-reimplementation/trans/optimal_expert_substitutions.py).

At a training step, the expert sees:

```text
source input
gold target
current source alignment
partial prediction so far
```

It then scores possible next actions:

```text
COPY
DELETE
INSERT(symbol)
SUBSTITUTE(symbol)
END
```

For each candidate action, it asks: after taking this action, what is the SED
cost of completing the remaining source/target suffixes?

The best-scoring action or actions become the optimal supervision for that
decoder step.

## Precomputing expert supervision

`precompute_from_expert()` in [trans/train.py](/Users/siclemat/pj/2023/il-reimplementation/trans/train.py) rolls out the expert before training.

For each sample it stores:

```text
action_history
alignment_history
optimal_actions_mask
valid_actions_mask
```

These fields are attached to each `Sample` object and later batched by
`Dataset.get_data_loader()`.

This matters for speed. Calling SED/expert logic inside every neural training
step would be expensive. Precomputation turns the expert output into tensors
that the neural model can train against directly.

## Neural training objective

The neural transducer predicts logits over the action vocabulary. Invalid
actions are masked out. The loss then maximizes probability mass assigned to
all expert-optimal actions.

This is different from ordinary single-label cross entropy. Sometimes multiple
actions are equally good according to the expert. The implementation preserves
that ambiguity with `optimal_actions_mask`.

The relevant method is:

```python
Transducer.log_sum_softmax_loss(...)
```

in [trans/transducer.py](/Users/siclemat/pj/2023/il-reimplementation/trans/transducer.py).

## Example: simple copy

Source:

```text
a
```

Target:

```text
a
```

A cheap path is:

```text
COPY a
END
```

or equivalently as an unconditional edit:

```text
Sub(a, a)
EOS
```

The expert will usually prefer copying when the current source symbol matches
the next target symbol.

## Example: substitution

Source:

```text
casa
```

Target:

```text
k a s a
```

At source position `c`, a likely action is:

```text
SUBSTITUTE(k)
```

because Italian `c` can map to `k` in some contexts. SED does not know the
contextual rule by itself, but it can learn from data that `c -> k` is a
plausible edit.

The neural model is responsible for learning the context-dependent pattern.
SED only provides action costs for supervision.

## Example: insertion

Source:

```text
ragazzo
```

Target:

```text
r a ɡ a t͡s o
```

Some target symbols may not align one-to-one with source symbols. An insertion
lets the output emit an extra target symbol while keeping the source alignment
fixed.

Conceptually:

```text
consume z -> emit t
insert ͡
insert s
```

The exact segmentation depends on the learned edit parameters and available
action vocabulary.

## Example: deletion

Source:

```text
hanno
```

Target:

```text
a n n o
```

If `h` is silent in the data, a plausible action is:

```text
DELETE h
```

The source alignment advances, but no target symbol is emitted.

## Valid actions vs optimal actions

The model distinguishes validity from optimality.

Valid actions are mechanically legal:

- `END` is always valid.
- insertions are valid even at the end of the input.
- copy/delete/substitute are valid only when input remains.

Optimal actions are what the expert recommends among valid actions.

For example, at the end of the source string:

```text
valid: END, INSERT(symbol)
invalid: COPY, DELETE, SUBSTITUTE(symbol)
```

Among those, the expert may prefer `END` if the prediction is complete, or an
insertion if target material is still missing.

## Important implementation files

- [trans/sed.py](/Users/siclemat/pj/2023/il-reimplementation/trans/sed.py):
  SED parameters, forward/backward algorithms, EM fitting, Viterbi alignment.
- [trans/optimal_expert_substitutions.py](/Users/siclemat/pj/2023/il-reimplementation/trans/optimal_expert_substitutions.py):
  expert policy that scores next actions using the SED aligner.
- [trans/train.py](/Users/siclemat/pj/2023/il-reimplementation/trans/train.py):
  SED loading/fitting and expert precomputation.
- [trans/transducer.py](/Users/siclemat/pj/2023/il-reimplementation/trans/transducer.py):
  neural encoder-decoder model, valid-action masking, training loss, greedy and
  beam decoding.
- [trans/utils.py](/Users/siclemat/pj/2023/il-reimplementation/trans/utils.py):
  `Sample`, `Dataset`, batching, and tensor device transfer.

## SED as a dataset diagnostic

A fitted SED model can also score how surprising each source-target pair is.
The command-line tool is:

```bash
trans-analyze-sed \
  --sed-params data.d/sed-2021/low_ita/sed.pkl \
  --input data.d/sigmorphon2021/low/ita_train.tsv \
  --output data.d/ita_train_sed_analysis.tsv
```

It writes a ranked TSV table containing:

```text
stochastic_surprisal      -log P(source,target), summed over alignments
max_length_surprisal      stochastic surprisal / max(source length, target length)
target_length_surprisal   stochastic surprisal / target length
viterbi_surprisal         -log P(best alignment, source,target)
alignment_ambiguity       log P(total) - log P(best alignment)
alignment                 readable Viterbi alignment
```

Useful rankings:

```bash
# Most surprising examples per output symbol.
trans-analyze-sed \
  --sed-params data.d/sed-2021/low_ita/sed.pkl \
  --input data.d/sigmorphon2021/low/ita_train.tsv \
  --sort-by target_length_surprisal \
  --limit 50

# Examples with many plausible alignments.
trans-analyze-sed \
  --sed-params data.d/sed-2021/low_ita/sed.pkl \
  --input data.d/sigmorphon2021/low/ita_train.tsv \
  --sort-by alignment_ambiguity \
  --limit 50
```

Raw surprisal tends to rank long examples highly, so normalized columns are
usually more useful for data cleaning. For principled anomaly detection, score
held-out examples with a SED model fitted on other data; scoring the same data
used to fit SED is still useful for exploratory inspection, but unusual examples
can partly teach the model their own edit behavior.

## Mental model

The cleanest way to understand this project is:

```text
SED learns edit costs from unaligned pairs.
The expert turns those costs into action-level supervision.
The neural transducer learns to imitate the expert.
At inference, only the neural transducer is used.
```

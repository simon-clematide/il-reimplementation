# Encoder Dropout Implementation Plan

## Goal

Add explicit, opt-in dropout on the LSTM encoder output representation without
changing the historical meaning of `--enc-dropout`. The implementation exposes
locked dropout directly because it is the more relevant sequence-consistent
regularizer for the one-layer BiLSTM experiments.

The key compatibility rule is:

- `--enc-dropout` remains the legacy PyTorch LSTM inter-layer dropout option.
- `--enc-output-dropout` is a new dropout applied to the full LSTM output
  representation before the transducer decoder consumes it.
- `--enc-output-dropout-type {none,standard,locked}` controls the output
  dropout variant; the default is `locked`.

This preserves old one-layer LSTM experiments where `--enc-dropout > 0` had no
effective dropout operation, because PyTorch applies LSTM dropout only between
stacked recurrent layers.

## Current State

`trans/encoders.py` currently defines:

```python
class LSTMEncoder(torch.nn.LSTM):
    ...
    super().__init__(
        input_size=args.char_dim,
        hidden_size=args.enc_hidden_dim,
        num_layers=args.enc_layers,
        bidirectional=args.enc_bidirectional,
        dropout=args.enc_dropout,
        device=args.device
    )
```

For `enc_layers == 1`, PyTorch accepts `dropout=args.enc_dropout` but does not
apply dropout in the recurrent stack. This behavior should remain intact for
reproducibility.

`Transducer.encoder_step()` receives the encoder output and then drops the
begin-word position:

```python
bidirectional_emb, _ = self.enc(input_emb)
return bidirectional_emb[1:]
```

The new output dropout should live inside `LSTMEncoder.forward()`, so it is
applied before the begin-word position is removed.

## Stage 1: Add Explicit Locked Output Dropout

Implementation:

1. Keep the existing `torch.nn.LSTM` constructor arguments unchanged.
2. Add new CLI arguments to `LSTMEncoder.add_args()`:

   ```text
   --enc-output-dropout FLOAT
   --enc-output-dropout-type {none,standard,locked}
   ```

   Defaults: `0.0` and `locked`.

3. Store an output dropout module:

   ```python
   self.output_dropout = LockedDropout(args.enc_output_dropout)
   ```

4. Override `LSTMEncoder.forward()`:

   ```python
   def forward(self, input, hx=None):
       output, state = super().forward(input, hx)
       output = self.output_dropout(output)
       return output, state
   ```

5. Do not map `enc_dropout` onto `enc_output_dropout` for one-layer encoders.

Acceptance tests:

- `enc_output_dropout=0.0` reproduces legacy output under controlled weights and
  fixed inputs.
- `enc_layers=1` with `enc_dropout > 0` alone does not change outputs across
  repeated training-mode forward passes.
- `enc_layers=1` with `enc_output_dropout > 0` changes training-mode outputs
  across repeated forward passes.
- `enc_output_dropout > 0` is inactive in `eval()` mode.
- Output shape remains `[T, B, D]`.
- Locked dropout mask is constant over `T`.

## Stage 2: Add Clear Warnings

Implementation:

1. Emit a warning when:

   ```text
   enc_layers == 1 and enc_dropout > 0
   ```

2. Warning text should explain:

   ```text
   enc-dropout has no effect for a one-layer LSTM; PyTorch applies LSTM dropout
   only between recurrent layers. Use --enc-output-dropout for explicit dropout
   on the encoder representation.
   ```

3. Show this warning even if `enc_output_dropout > 0`, because the two options
   have different semantics.

Acceptance tests:

- Constructing a one-layer LSTM encoder with `enc_dropout > 0` emits the warning.
- Constructing a one-layer LSTM encoder with `enc_dropout == 0` does not emit
  the warning.
- Constructing a two-layer LSTM encoder with `enc_dropout > 0` does not emit
  the warning.

## Stage 3: CLI And Documentation

Implementation:

1. Document `--enc-output-dropout` in the encoder CLI help.
2. Update project documentation to distinguish:

   - `enc_dropout`: PyTorch LSTM inter-layer dropout.
   - `enc_output_dropout`: dropout on the encoder output sequence.

3. Add a grid-search example for the first controlled experiment:

   ```json
   {
     "runs_per_model": 3,
     "seeds": [1, 2, 3],
     "grids": {
       "one_layer_output_dropout": {
         "enc-type": "lstm",
         "enc-layers": 1,
         "enc-dropout": 0.0,
         "enc-output-dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
         "enc-output-dropout-type": "locked"
       }
     }
   }
   ```

Acceptance tests:

- Parser defaults include `enc_output_dropout == 0.0` and
  `enc_output_dropout_type == "locked"`.
- Parser accepts a positive float value.
- `trans-train --help` exposes `--enc-output-dropout`.
- `trans-train --help` exposes `--enc-output-dropout-type`.

## Stage 4: Controlled Experiment

Use the new seeded grid-search machinery and hold the successful one-layer
architecture fixed.

Recommended first grid:

```json
{
  "runs_per_model": 3,
  "seeds": [1, 2, 3],
  "grids": {
    "one_layer_output_dropout": {
      "enc-type": "lstm",
      "enc-layers": 1,
      "enc-dropout": 0.0,
      "enc-output-dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
      "enc-output-dropout-type": "locked"
    }
  }
}
```

Question:

```text
Does explicit representation dropout improve the strong one-layer LSTM model?
```

Report:

- mean ± sample SD across the same replicate seeds
- dev greedy
- dev beam, if beam decoding is enabled
- test results only after selecting on dev

Avoid simultaneously grid-searching hidden size, learning rate, depth, and
dropout in this first experiment. The goal is to isolate the regularizer.

## Stage 5: Implemented Locked Dropout

Locked dropout is implemented as the default output-dropout type.

Design:

```text
--enc-output-dropout FLOAT
--enc-output-dropout-type {standard,locked}
```

For an encoder output with shape `[T, B, D]`, locked dropout uses a mask of
shape `[1, B, D]` and broadcasts it over time. This removes an encoder feature
dimension for a whole sequence/example rather than independently at every time
step.

Acceptance tests:

- Locked dropout mask is constant over `T`.
- Different batch examples may receive different masks.
- Dropout remains inactive in `eval()`.
- `enc_output_dropout=0.0` behaves identically for both dropout types.

## Stage 6: Later MC-Dropout Decoding

Do not implement MC dropout in the initial patch.

If output dropout improves individual models, future work can evaluate
stochastic inference by keeping dropout active at prediction time and combining
multiple decoded sequences. This requires a separate design because the model
outputs structured edit-action sequences, not independent class probabilities.

## Non-Goals

- Do not change the semantics of `--enc-dropout`.
- Do not silently activate output dropout when `enc_layers == 1`.
- Do not claim ordinary dropout is equivalent to independently trained
  ensembles.
- Do not implement MC dropout in the first patch.

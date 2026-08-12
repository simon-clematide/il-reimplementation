# Bug Fixes and Testability Plan

## Scope and priority

This plan covers the Python package and the G2P notebook. It is based on a
static review of the execution paths plus an attempted test run. The current
environment cannot import the test suite because the package dependencies are
not installed; this directly corroborates the packaging issue below.

The ordering is intentional: the first phase restores a reproducible runtime;
the next two phases correct results and prevent data-dependent failures. Each
change must land with its regression tests.

### P0: installation and notebook reproducibility

1. Make `pyproject.toml` the sole authoritative package definition.

   The active dependency list contains only `Cython`, although importing or
   running the package needs at least PyTorch, NumPy, SciPy, `editdistance`,
   and `progressbar`. Modern `pip` builds from `pyproject.toml`, so the G2P
   notebook's Git install does not reliably create a usable runtime. It also
   references a `LICENSE` file that is absent from the repository.

   Implementation:
   - Define supported Python versions and compatible, unpinned-or-bounded
     runtime dependencies in `pyproject.toml`.
   - Remove duplicated or conflicting dependency metadata from `setup.py`, or
     reduce it to a compatibility shim that reads the same metadata.
   - Add the declared license file or correct the metadata.
   - Update `requirements.txt` only if it becomes a deliberately maintained
     locked environment; otherwise remove its role ambiguity.
   - Update the notebook install cell to install a released/tagged revision or
     a clearly documented branch, then restart/import-check the runtime.

   Tests / acceptance:
   - Build a wheel in a clean virtual environment and install it.
   - Run `python -m unittest discover -s trans/tests -v` after installation.
   - Smoke-test `trans-train --help`, `trans-ensemble --help`, and
     `trans-grid-search --help`.
   - Execute the notebook's setup and a minimal one-epoch CPU G2P example in a
     fresh kernel.

### P1: training and model correctness

2. Correct gradient accumulation in `trans/train.py`.

   Batches are stepped when `j % grad_accumulation == 0`, which steps the very
   first batch and then every `grad_accumulation` batches thereafter. A final
   incomplete group is never stepped. This changes optimization substantially
   whenever accumulation is greater than one.

   Implementation:
   - Extract a small `should_step(batch_index, batch_count, accumulation)`
     helper, or use `(batch_index + 1) % accumulation == 0` plus an explicit
     final-batch condition.
   - Scale a partial final group by its actual number of batches, rather than
     always by `grad_accumulation`, when using mean reduction.
   - Call `optimizer.zero_grad(set_to_none=True)` before the loop and after
     each step.

   Tests / acceptance:
   - Unit-test optimizer step counts for 1, 2, 3, and 5 batches with an
     accumulation factor of 2.
   - Verify that parameters after accumulated micro-batches match an equivalent
     reference update on the combined batch, within floating-point tolerance.
   - Cover the trailing partial group explicitly.

3. Remove the fixed 100-character decoding limit in `trans/transducer.py`.

   `valid_actions_lookup` has entries only for suffix lengths 0 through 99,
   while decoding indexes `len(input) + 1 - alignment`. A 99-character input
   therefore indexes 100 and raises `IndexError`; the effective safe limit is
   lower than the named constant suggests.

   Implementation:
   - Replace the fixed lookup table with on-demand masks, preferably cached by
     suffix length in a normal dictionary, or size the table from the current
     batch's maximum encoded length.
   - Use the same helper from greedy decoding, beam decoding, and expert
     precomputation so their validity rules cannot diverge.
   - Keep a separate, user-configurable action-output safety limit, with a
     clear error or truncated-status result when reached.

   Tests / acceptance:
   - Greedy and beam decoding accept inputs of length 0, 1, 99, 100, and 250.
   - Assert that each returned action is valid at the corresponding alignment.
   - Include a batch whose members have substantially different lengths.

4. Fix transformer positional embeddings for batches.

   `SinusoidalPositionalEmbedding.forward()` receives a batch-by-sequence
   padding mask, but computes `torch.cumsum(mask, dim=0)`. Positions therefore
   accumulate across examples in a batch instead of across tokens within each
   example. Transformer results depend on batch composition and are incorrect
   for batch sizes above one.

   Implementation:
   - Accumulate on the sequence axis (`dim=1`) and preserve the expected
     batch-first positional embedding layout.
   - Add shape assertions or type annotations to distinguish batch-first masks
     from sequence-first encoder tensors.
   - Ensure padding positions remain the zero embedding.

   Tests / acceptance:
   - Identical unpadded token positions in two batch rows receive identical
     positional vectors.
   - Adding a second, differently padded example does not alter the first
     example's positions or transformer encoder output in evaluation mode.
   - Test odd and even embedding dimensions and dynamic expansion past the
     initial positional-table length.

5. Guarantee that a checkpoint is always produced and is portable.

   `best_dev_accuracy` starts at zero and the checkpoint is saved only for a
   strictly greater score. A valid run whose first and all subsequent scores
   are zero has no `best.model`, then fails when reload is attempted. Loading
   also lacks `map_location`, making GPU-created checkpoints awkward on CPU.

   Implementation:
   - Initialize the best score to negative infinity or save the first epoch
     unconditionally; validate that at least one epoch ran before reload.
   - Save run metadata beside the state dict: package version/commit, CLI
     arguments, vocabulary, and device-independent tensors.
   - Load with `map_location=args.device`.

   Tests / acceptance:
   - Stub decoding to return zero accuracy and assert a checkpoint and final
     prediction files are still produced.
   - Save on CPU and reload using an explicit CPU map location.

6. Move all precomputed sample tensors between devices.

   `Dataset.to()` omits `encoded_features`, so loading precomputed
   feature-based data to CUDA leaves feature IDs on CPU and fails in the
   embedding lookup.

   Implementation:
   - Include `encoded_features` in the migrated attributes.
   - Centralize the tensor field list on `Sample` or expose a `Sample.to()`
     method to avoid future omissions.

   Tests / acceptance:
   - Construct a precomputed sample with every tensor field, move it to the
     selected device, and assert each tensor's device.
   - Run a feature-enabled `training_step`; make the CUDA assertion conditional
     on `torch.cuda.is_available()`.

### P2: orchestration and alignment correctness

7. Repair `trans/grid_search.py` command construction and process handling.

   The ensemble command misses a comma, producing the executable name
   `trans-ensemble--gold`. False boolean configuration values are emitted as
   `--flag False`, which breaks `store_true` flags such as `--nfd`. The local
   `process_list` shadows the global list inspected by `cleanup()`, so SIGINT
   cleanup does not terminate children. Existing output directories also make
   a rerun fail, and child exit statuses are ignored.

   Implementation:
   - Extract pure `build_train_command()` and `build_ensemble_command()`
     functions that return argument lists.
   - Omit false action flags; use explicit value-taking options only where the
     parser supports them.
   - Use one owned process manager (not a global/local shadow), terminate and
     wait for active children in `finally`, and surface command, return code,
     and log path on failure.
   - Decide and document rerun semantics: reject nonempty output by default,
     or add an explicit resume/overwrite option. Do not silently mix results.
   - Use context managers for configuration files and `pathlib.Path` for paths.

   Tests / acceptance:
   - Unit-test exact command arrays, including true/false flags, multi-value
     arguments, and the ensemble executable token.
   - Mock `Popen` to verify nonzero exit status stops aggregation and cleanup
     terminates active processes.
   - Test empty, existing-empty, and existing-nonempty output directories.

8. Correct SED Viterbi boundary backtracking and empty-string handling.

   In `viterbi_distance(..., with_alignment=True)`, boundary updates subtract
   an index from itself rather than decrementing it. For leading insertions or
   deletions, backtracking jumps straight to zero and loses edits. In `e_step`,
   source and target are indexed before the `t > 0` / `v > 0` guards, so empty
   strings can raise `IndexError`.

   Implementation:
   - Decrement boundary indices by one before creating the corresponding edit.
   - Move source/target indexing inside guarded branches.
   - Add an alignment replay helper used by tests to ensure edits transform the
     source into the target.

   Tests / acceptance:
   - Exact alignment cases for `"" -> "abc"`, `"abc" -> ""`, and mixed
     prefix/suffix insertions and deletions.
   - EM update and likelihood tests containing empty source and target strings.
   - Assert alignment score equals the Viterbi dynamic-programming score.

### P3: CLI and maintainability fixes

9. Replace every `argparse` `type=bool` option.

   `bool("False")` is true, so `--enc-bidirectional False` and optimizer
   boolean options do not mean what users expect. Use `BooleanOptionalAction`
   for modern supported Python versions, or paired `--foo` / `--no-foo` flags
   for compatibility.

10. Make decoding and batching interfaces explicit.

   Remove the unused `input_tensor` in `input_embedding()`, validate feature
   presence when the model is configured with features, and replace opaque
   tensor reshaping/index construction in `decoder_step()` with named shape
   variables. These are not the highest-risk defects, but they are worthwhile
   while adding the tests above.

## Test architecture

Keep `unittest` unless the project intentionally adopts `pytest`; the immediate
need is coverage, not a framework migration. Add focused files rather than
expanding only end-to-end tests:

- `test_train_loop.py`: accumulation helper, checkpoint selection, and a tiny
  CPU one-epoch integration test using a temporary TSV corpus.
- `test_decoding.py`: dynamic valid-action masks, greedy/beam decoding bounds,
  output-limit behavior, and action validity invariants.
- `test_encoders.py`: positional embedding values, padding, batching isolation,
  and transformer forward shapes.
- `test_dataset.py`: collation with/without features, padding, serialization,
  and device transfer.
- `test_grid_search.py`: pure command construction and mocked process failure /
  cleanup behavior.
- Extend `test_sed.py`: boundary alignments, alignment replay, and empty input.
- `test_cli.py`: parser defaults and positive/negative boolean spelling without
  invoking a full training job.

Use small deterministic fixtures: a fixed vocabulary, 1-3 examples, seeded
PyTorch, dropout disabled, and temporary directories. Mock only process
creation, filesystem failure paths, and expensive decoding; keep tensor,
alignment, and parser behavior real. Add a CPU test job as the required CI
baseline and a CUDA smoke job only when a GPU runner is available.

## Re-checked prominence

The original headline issues remain prominent: packaging, accumulation,
checkpoint creation, decoding bounds, feature device migration, grid-search
command construction, boolean parsing, and SED alignment are all confirmed.
Two findings should be promoted into the top implementation phase: transformer
positional indexing is a direct model-correctness defect for batches, and
grid-search process cleanup is unreliable. Conversely, the unused local in
`input_embedding()` and general path/file-handle cleanup are real but should
not delay the correctness work.

The first practical milestone is P0 plus P1 items 2, 4, and 5: after that the
notebook can be installed cleanly and a basic CPU run has trustworthy training,
batching, and checkpoint semantics. Then implement dynamic decoding and the
device fix before relying on longer G2P inputs, CUDA, or feature-based data.

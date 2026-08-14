"""Unit tests for train-loop helpers."""
import argparse
import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from trans.actions import ConditionalIns
from trans import transducer
from trans import train
from trans import utils
from trans import vocabulary


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class TestGradientAccumulation(unittest.TestCase):

    @staticmethod
    def small_transducer_args(device="cpu"):
        return argparse.Namespace(
            device=device,
            char_dim=4,
            action_dim=4,
            enc_type="lstm",
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=0,
            expert_temperature=0.,
            expert_loss="marginal",
            expert_margin=1.0,
            dec_hidden_dim=4,
            dec_layers=1,
        )

    @staticmethod
    def small_vocabularies():
        vocabularies = vocabulary.Vocabularies(characters=["a"])
        vocabularies.encode_actions(["a"])
        return vocabularies

    @staticmethod
    def run_accumulated_updates(model, micro_batches, accumulation):
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        batch_count = len(micro_batches)
        for i, (x, y) in enumerate(micro_batches):
            prediction = model(x)
            losses = (prediction - y).pow(2).mean(dim=1)
            scale = train.accumulation_loss_scale(i, batch_count, accumulation)
            (torch.mean(losses) / scale).backward()
            if train.should_step(i, batch_count, accumulation):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return optimizer

    @staticmethod
    def run_reference_updates(model, groups):
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            x = torch.cat([batch[0] for batch in group], dim=0)
            y = torch.cat([batch[1] for batch in group], dim=0)
            prediction = model(x)
            loss = (prediction - y).pow(2).mean()
            loss.backward()
            optimizer.step()
        return optimizer

    @staticmethod
    def linear_model():
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5)
        return model

    @staticmethod
    def micro_batches():
        return [
            (torch.tensor([[1.0]]), torch.tensor([[2.0]])),
            (torch.tensor([[2.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[3.0]]), torch.tensor([[0.0]])),
            (torch.tensor([[4.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[5.0]]), torch.tensor([[3.0]])),
        ]

    def test_should_step_every_accumulation_and_final_batch(self):
        cases = {
            1: [True],
            2: [False, True],
            3: [False, True, True],
            5: [False, True, False, True, True],
        }
        for batch_count, expected in cases.items():
            with self.subTest(batch_count=batch_count):
                actual = [
                    train.should_step(i, batch_count, accumulation=2)
                    for i in range(batch_count)
                ]
                self.assertEqual(expected, actual)

    def test_accumulation_loss_scale_handles_partial_final_group(self):
        actual = [
            train.accumulation_loss_scale(i, batch_count=5, accumulation=2)
            for i in range(5)
        ]
        self.assertEqual([2, 2, 2, 2, 1], actual)

    def test_invalid_accumulation_is_rejected(self):
        with self.assertRaises(ValueError):
            train.should_step(0, 1, 0)
        with self.assertRaises(ValueError):
            train.accumulation_loss_scale(0, 1, 0)

    def test_accumulation_steps_trailing_partial_group(self):
        model = self.linear_model()

        optimizer = self.run_accumulated_updates(
            model, self.micro_batches(), accumulation=2)

        self.assertEqual(3, optimizer.step_count)

    def test_accumulated_updates_match_grouped_reference_updates(self):
        micro_batches = self.micro_batches()
        accumulated_model = self.linear_model()
        reference_model = self.linear_model()

        accumulated_optimizer = self.run_accumulated_updates(
            accumulated_model, micro_batches, accumulation=2)
        reference_optimizer = self.run_reference_updates(
            reference_model,
            [micro_batches[0:2], micro_batches[2:4], micro_batches[4:5]],
        )

        self.assertEqual(reference_optimizer.step_count,
                         accumulated_optimizer.step_count)
        self.assertTrue(torch.allclose(
            reference_model.weight,
            accumulated_model.weight,
        ))

    def test_write_checkpoint_metadata(self):
        args = argparse.Namespace(device="cpu", epochs=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "best.model.json")

            train.write_checkpoint_metadata(
                metadata_path,
                args,
                epoch=0,
                dev_string_accuracy=0.0,
                dev_symbol_accuracy=0.5,
                train_string_accuracy=0.25,
            )

            with open(metadata_path) as f:
                metadata = json.load(f)

        self.assertEqual(0, metadata["epoch"])
        self.assertEqual(0.0, metadata["dev_string_accuracy"])
        self.assertEqual(0.5, metadata["dev_symbol_accuracy"])
        self.assertEqual(0.25, metadata["train_string_accuracy"])
        self.assertEqual({"device": "cpu", "epochs": 1}, metadata["args"])
        self.assertIn("git_commit", metadata)

    def test_levenshtein_distance(self):
        self.assertEqual(0, train.levenshtein_distance(["a"], ["a"]))
        self.assertEqual(1, train.levenshtein_distance(["a"], ["b"]))
        self.assertEqual(1, train.levenshtein_distance(["a"], ["a", "b"]))
        self.assertEqual(2, train.levenshtein_distance(["a", "b"], ["c", "d"]))

    def test_output_symbols_handles_empty_separated_output(self):
        self.assertEqual([], train.output_symbols("", utils.Tokenizer(" ")))
        self.assertEqual(["d͡ʒ", "a"], train.output_symbols("d͡ʒ a", utils.Tokenizer(" ")))

    def test_model_selection_key_prefers_string_then_symbol_accuracy(self):
        self.assertGreater(
            train.model_selection_key(0.8, 0.5),
            train.model_selection_key(0.7, 0.0),
        )
        self.assertGreater(
            train.model_selection_key(0.8, 0.9),
            train.model_selection_key(0.8, 0.8),
        )

    def test_load_transducer_for_device_uses_requested_device(self):
        vocabularies = self.small_vocabularies()
        args = self.small_transducer_args(device="cpu")
        model = transducer.Transducer(vocabularies, None, args)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)

            loaded = train.load_transducer_for_device(
                vocabularies,
                None,
                args,
                model_path,
                "cpu",
            )

        self.assertEqual(torch.device("cpu"), loaded.device)
        self.assertFalse(loaded.training)

    def test_write_sed_metadata(self):
        args = argparse.Namespace(
            train="train.tsv",
            source_separator=None,
            target_separator=" ",
            sed_em_iterations=3,
            sed_em_mode="damped",
            sed_em_damping=0.9,
        )
        vocabularies = vocabulary.Vocabularies(
            characters=["a", "b"],
            source_separator=None,
            target_separator=" ",
        )
        vocabularies.encode_actions(["a", "d͡ʒ"])
        dataset = utils.Dataset([
            utils.Sample(["a"], ["a"]),
            utils.Sample(["b"], ["d͡ʒ"]),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "sed.pkl.json")

            train.write_sed_metadata(
                metadata_path,
                args,
                dataset,
                vocabularies,
            )

            with open(metadata_path) as f:
                metadata = json.load(f)

        self.assertEqual("sed.pkl", metadata["sed_params"])
        self.assertEqual("train.tsv", metadata["train"])
        self.assertIsNone(metadata["source_separator"])
        self.assertEqual(" ", metadata["target_separator"])
        self.assertEqual(3, metadata["em_iterations"])
        self.assertEqual(2, metadata["num_samples"])
        self.assertIn("d͡ʒ", metadata["target_alphabet"])

    def test_precompute_from_expert_records_output_history(self):
        vocabularies = vocabulary.Vocabularies(characters=["a"])
        vocabularies.encode_actions(["a"])
        model_args = argparse.Namespace(
            device="cpu",
            char_dim=4,
            action_dim=4,
            enc_type="lstm",
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=2,
            expert_temperature=0.,
            expert_loss="marginal",
            dec_hidden_dim=4,
            dec_layers=1,
        )
        transducer_ = transducer.Transducer(
            vocabularies,
            None,
            model_args,
        )
        action_script = iter([
            {transducer_.vocab.decode_action(vocabulary.COPY): 0.},
            {transducer_.vocab.decode_action(vocabulary.END_WORD): 0.},
        ])
        transducer_.expert_action_scores = lambda input_, target, alignment, output: next(action_script)
        sample = utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1]))

        train.precompute_from_expert(sample, transducer_)

        expected = torch.tensor([
            vocabularies.encode_output_symbol(vocabulary.BOS_OUTPUT),
            vocabularies.encode_output_symbol("a"),
        ])
        self.assertTrue(torch.equal(expected, sample.output_history))
        self.assertEqual(
            (2, transducer_.number_actions),
            tuple(sample.expert_action_costs.shape),
        )
        self.assertEqual(torch.device("cpu"), sample.optimal_actions_mask.device)
        self.assertEqual(torch.device("cpu"), sample.alignment_history.device)
        self.assertEqual(torch.device("cpu"), sample.action_history.device)
        self.assertEqual(torch.device("cpu"), sample.output_history.device)
        self.assertEqual(torch.device("cpu"), sample.expert_action_costs.device)
        self.assertEqual(torch.device("cpu"), sample.valid_actions_mask.device)

    def test_precompute_from_expert_rejects_unknown_expert_action_with_context(self):
        vocabularies = vocabulary.Vocabularies(characters=["a"])
        vocabularies.encode_actions(["a"])
        model_args = argparse.Namespace(
            device="cpu",
            char_dim=4,
            action_dim=4,
            enc_type="lstm",
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=0,
            expert_temperature=0.,
            expert_loss="marginal",
            dec_hidden_dim=4,
            dec_layers=1,
        )
        transducer_ = transducer.Transducer(
            vocabularies,
            None,
            model_args,
        )
        transducer_.expert_action_scores = lambda input_, target, alignment, output: {
            ConditionalIns("z"): 0.,
        }
        sample = utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1]))

        with self.assertRaisesRegex(
                RuntimeError,
                "precompute_from_expert: input=\\['a'\\].*absent from action vocabulary"):
            train.precompute_from_expert(sample, transducer_)

    def test_should_refresh_rollin(self):
        self.assertFalse(train.should_refresh_rollin(4, start=5, refresh=5, probability=0.2))
        self.assertTrue(train.should_refresh_rollin(5, start=5, refresh=5, probability=0.2))
        self.assertFalse(train.should_refresh_rollin(6, start=5, refresh=5, probability=0.2))
        self.assertTrue(train.should_refresh_rollin(10, start=5, refresh=5, probability=0.2))
        self.assertFalse(train.should_refresh_rollin(10, start=5, refresh=5, probability=0.0))

    def test_focal_gamma_schedule(self):
        self.assertEqual(0., train.focal_gamma_schedule(
            epoch=4,
            target_gamma=1.,
            start=5,
            ramp=15,
        ))
        self.assertAlmostEqual(1. / 15, train.focal_gamma_schedule(
            epoch=5,
            target_gamma=1.,
            start=5,
            ramp=15,
        ))
        self.assertEqual(1., train.focal_gamma_schedule(
            epoch=19,
            target_gamma=1.,
            start=5,
            ramp=15,
        ))
        self.assertEqual(2., train.focal_gamma_schedule(
            epoch=5,
            target_gamma=2.,
            start=5,
            ramp=0,
        ))

    def test_precompute_rollin_executes_model_action_but_stores_expert_supervision(self):
        vocabularies = vocabulary.Vocabularies(characters=["a"])
        vocabularies.encode_actions(["a"])
        model_args = argparse.Namespace(
            device="cpu",
            char_dim=4,
            action_dim=4,
            enc_type="lstm",
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=0,
            expert_temperature=0.,
            expert_loss="marginal",
            dec_hidden_dim=4,
            dec_layers=1,
        )
        transducer_ = transducer.Transducer(
            vocabularies,
            None,
            model_args,
        )

        copy_action = transducer_.vocab.decode_action(vocabulary.COPY)
        delete_action = transducer_.vocab.decode_action(vocabulary.DELETE)
        end_action = transducer_.vocab.decode_action(vocabulary.END_WORD)

        def expert_action_scores(input_, target, alignment, output):
            if alignment == 0:
                return {copy_action: 0., delete_action: 1.}
            return {end_action: 0.}

        original_rollin_action = train.model_greedy_rollin_action
        transducer_.expert_action_scores = expert_action_scores
        model_actions = iter([vocabulary.DELETE, vocabulary.END_WORD])
        train.model_greedy_rollin_action = \
            lambda sample, model, alignments, actions, outputs: next(model_actions)
        sample = utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1]))
        try:
            stats = train.precompute_from_expert(
                sample,
                transducer_,
                device="cpu",
                rollin_prob=1.0,
                rollin_policy="greedy",
                rollin_rng=random.Random(1),
            )
        finally:
            train.model_greedy_rollin_action = original_rollin_action

        self.assertEqual(vocabulary.DELETE, sample.action_history[1].item())
        self.assertTrue(sample.optimal_actions_mask[0, vocabulary.COPY])
        self.assertFalse(sample.optimal_actions_mask[0, vocabulary.DELETE])
        self.assertEqual(2, stats.model_controlled)
        self.assertEqual(1, stats.model_non_optimal)
        self.assertEqual(1, stats.model_expert_agree)

    def test_precompute_rollin_uses_dedicated_rng(self):
        vocabularies = vocabulary.Vocabularies(characters=["a"])
        vocabularies.encode_actions(["a"])
        model_args = argparse.Namespace(
            device="cpu",
            char_dim=4,
            action_dim=4,
            enc_type="lstm",
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=0,
            expert_temperature=0.,
            expert_loss="marginal",
            dec_hidden_dim=4,
            dec_layers=1,
        )
        transducer_ = transducer.Transducer(vocabularies, None, model_args)
        copy_action = transducer_.vocab.decode_action(vocabulary.COPY)
        end_action = transducer_.vocab.decode_action(vocabulary.END_WORD)
        transducer_.expert_action_scores = lambda input_, target, alignment, output: \
            {copy_action: 0.} if alignment == 0 else {end_action: 0.}
        original_random = random.random
        original_rollin_action = train.model_greedy_rollin_action
        model_actions = iter([vocabulary.COPY, vocabulary.END_WORD])
        random.random = lambda: (_ for _ in ()).throw(AssertionError("global RNG used"))
        train.model_greedy_rollin_action = \
            lambda sample, model, alignments, actions, outputs: next(model_actions)
        try:
            stats = train.precompute_from_expert(
                utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1])),
                transducer_,
                device="cpu",
                rollin_prob=1.0,
                rollin_policy="greedy",
                rollin_rng=random.Random(1),
            )
        finally:
            random.random = original_random
            train.model_greedy_rollin_action = original_rollin_action

        self.assertEqual(2, stats.model_controlled)

    def test_precompute_rollin_uses_dynamic_rollout_cap(self):
        vocabularies = vocabulary.Vocabularies(characters=["a"])
        vocabularies.encode_actions(["a"])
        model_args = argparse.Namespace(
            device="cpu",
            char_dim=4,
            action_dim=4,
            enc_type="lstm",
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=0,
            expert_temperature=0.,
            expert_loss="marginal",
            dec_hidden_dim=4,
            dec_layers=1,
        )
        transducer_ = transducer.Transducer(vocabularies, None, model_args)
        insert_action = transducer_.vocab.decode_action(
            transducer_.vocab.encode_unseen_action(ConditionalIns("a")))
        transducer_.expert_action_scores = \
            lambda input_, target, alignment, output: {insert_action: 0.}

        stats = train.precompute_from_expert(
            utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1])),
            transducer_,
            device="cpu",
        )

        self.assertEqual(1, stats.truncated)
        self.assertLess(stats.trajectory_lengths[0], transducer.MAX_ACTION_SEQ_LEN)

    def test_best_non_optimal_cost_gaps(self):
        dataset = utils.Dataset([
            utils.Sample(
                ["a"],
                ["a"],
                expert_action_costs=torch.tensor([
                    [0., 0.2, float("inf"), 1.0],
                    [3., 3., float("inf"), 4.5],
                    [2., float("inf"), float("inf"), float("inf")],
                ]),
            )
        ])

        self.assertTrue(np.allclose(
            [0.2, 1.5],
            train.best_non_optimal_cost_gaps(dataset),
        ))

    def test_log_expert_gap_statistics_reports_weight_coverage(self):
        dataset = utils.Dataset([
            utils.Sample(
                ["a"],
                ["a"],
                expert_action_costs=torch.tensor([
                    [0., 1., float("inf")],
                    [0., 4., float("inf")],
                ]),
            )
        ])

        with self.assertLogs(level="INFO") as logs:
            train.log_expert_gap_statistics(dataset, temperature=4.0)

        self.assertIn(
            "Best non-optimal expert weight coverage at tau=4.0000",
            "\n".join(logs.output),
        )

    def test_soft_expert_distribution_statistics(self):
        dataset = utils.Dataset([
            utils.Sample(
                ["a"],
                ["a"],
                expert_action_costs=torch.tensor([
                    [0., 2., float("inf")],
                    [1., 1., float("inf")],
                ]),
            )
        ])

        optimal_masses, entropies, effective_sizes = \
            train.soft_expert_distribution_statistics(dataset, temperature=2.0)

        first_probs = np.array([1.0, np.exp(-1.0)])
        first_probs = first_probs / first_probs.sum()
        first_entropy = -np.sum(first_probs * np.log(first_probs))
        self.assertTrue(np.allclose(
            [first_probs[0], 1.0],
            optimal_masses,
        ))
        self.assertTrue(np.allclose(
            [first_entropy, np.log(2.0)],
            entropies,
        ))
        self.assertTrue(np.allclose(
            [np.exp(first_entropy), 2.0],
            effective_sizes,
        ))

    def test_log_expert_gap_statistics_reports_soft_expert_summary(self):
        dataset = utils.Dataset([
            utils.Sample(
                ["a"],
                ["a"],
                expert_action_costs=torch.tensor([
                    [0., 1., float("inf")],
                ]),
            )
        ])

        with self.assertLogs(level="INFO") as logs:
            train.log_expert_gap_statistics(dataset, temperature=2.0)

        self.assertIn(
            "Normalized soft expert at tau=2.0000",
            "\n".join(logs.output),
        )

    def test_log_margin_statistics_reports_epoch_diagnostics(self):
        stats = {
            "states": 10,
            "active": 3,
            "margin_sum": 15.,
            "margin_count": 5,
            "active_loss_sum": 1.5,
        }

        with self.assertLogs(level="INFO") as logs:
            train.log_margin_statistics(stats)

        output = "\n".join(logs.output)
        self.assertIn("Margin active: 0.3000.", output)
        self.assertIn("Mean oracle/nonoracle logit margin: 3.0000.", output)
        self.assertIn("Mean active margin loss: 0.5000.", output)

    def test_should_stop_for_patience(self):
        self.assertFalse(train.should_stop_for_patience(0, 2))
        self.assertFalse(train.should_stop_for_patience(1, 2))
        self.assertTrue(train.should_stop_for_patience(2, 2))
        self.assertTrue(train.should_stop_for_patience(3, 2))

    def test_invalid_patience_is_rejected(self):
        with self.assertRaises(ValueError):
            train.should_stop_for_patience(0, 0)

    def test_optimizer_learning_rates_reports_all_param_groups(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(
            [
                {"params": [model.weight], "lr": 0.1},
                {"params": [model.bias], "lr": 0.01},
            ],
        )

        self.assertEqual([0.1, 0.01], train.optimizer_learning_rates(optimizer))

    def test_log_learning_rate_change_logs_actual_change(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = train.optimizer_learning_rates(optimizer)
        optimizer.param_groups[0]["lr"] = 0.05

        with self.assertLogs(level="INFO") as logs:
            train.log_learning_rate_change(before, optimizer, "reduce_on_plateau")

        self.assertIn(
            "Learning rate changed by reduce_on_plateau scheduler: [0.1] -> [0.05].",
            logs.output[0],
        )

    def test_log_learning_rate_change_is_quiet_without_change(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = train.optimizer_learning_rates(optimizer)

        with self.assertNoLogs(level="INFO"):
            train.log_learning_rate_change(before, optimizer, "reduce_on_plateau")


if __name__ == "__main__":
    unittest.main()

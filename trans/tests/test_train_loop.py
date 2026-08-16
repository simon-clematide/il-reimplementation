"""Unit tests for train-loop helpers."""
import argparse
import json
import logging
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from trans import optimal_expert_substitutions
from trans.actions import ConditionalCopy, ConditionalDel, ConditionalIns
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

    def test_configure_output_file_logging_persists_info_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_logger = logging.getLogger()
            before_handlers = list(root_logger.handlers)
            before_level = root_logger.level
            try:
                root_logger.setLevel(logging.INFO)
                log_path = train.configure_output_file_logging(tmpdir)
                duplicate_path = train.configure_output_file_logging(tmpdir)
                logging.info("persistent diagnostic message")
                for handler in root_logger.handlers:
                    handler.flush()

                file_handlers = [
                    handler
                    for handler in root_logger.handlers
                    if isinstance(handler, logging.FileHandler) and
                    os.path.abspath(handler.baseFilename) == os.path.abspath(log_path)
                ]
                with open(log_path) as f:
                    log_text = f.read()
            finally:
                root_logger.setLevel(before_level)
                for handler in root_logger.handlers[:]:
                    if handler not in before_handlers:
                        root_logger.removeHandler(handler)
                        handler.close()

        self.assertEqual(log_path, duplicate_path)
        self.assertEqual(1, len(file_handlers))
        self.assertIn("persistent diagnostic message", log_text)

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
        self.assertIsNone(metadata["copy_probability"])
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

    def test_precompute_critic_model_action_reports_without_augmenting_costs(self):
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
        copy_action = ConditionalCopy()
        delete_action = ConditionalDel()
        end_action = transducer_.vocab.decode_action(vocabulary.END_WORD)

        def expert_action_scores(input_, target, alignment, output):
            if alignment == 0:
                return {copy_action: 0.}
            return {end_action: 0.}

        transducer_.expert_action_scores = expert_action_scores
        transducer_.expert_score_action = \
            lambda input_, target, alignment, output, action_id: 1000.5
        def expert_score_decoder_action(input_, target, alignment, output, action_id):
            total = 2. if action_id == vocabulary.DELETE else 5.
            return optimal_expert_substitutions.DecoderStateScore(
                prefix_cost=1.,
                continuation_cost=total - 1.,
                total=total,
                target_prefix_index=1,
            )

        transducer_.expert_score_decoder_action = expert_score_decoder_action
        original_rollin_action = train.model_greedy_rollin_action
        model_actions = iter([vocabulary.DELETE, vocabulary.END_WORD])
        train.model_greedy_rollin_action = \
            lambda sample, model, alignments, actions, outputs: next(model_actions)
        sample = utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1]))
        try:
            stats = train.precompute_from_expert(
                sample,
                transducer_,
                device="cpu",
                critic_model_action=True,
            )
        finally:
            train.model_greedy_rollin_action = original_rollin_action

        self.assertTrue(sample.optimal_actions_mask[0, vocabulary.COPY])
        self.assertFalse(sample.optimal_actions_mask[0, vocabulary.DELETE])
        self.assertTrue(torch.isinf(sample.expert_action_costs[0, vocabulary.DELETE]))
        self.assertEqual(2, stats.critic_states)
        self.assertEqual(1, stats.critic_new_actions)
        self.assertEqual(0, stats.critic_better_actions)
        self.assertEqual(2., stats.critic_regret_sum)
        self.assertEqual([2.], stats.critic_finite_gaps)
        self.assertEqual([2.], stats.critic_gaps_by_action_type["DEL"])
        self.assertEqual([1000.5], stats.critic_old_costs_by_action_type["DEL"])
        self.assertEqual([1.], stats.critic_prefix_costs_by_action_type["DEL"])
        self.assertEqual([1.], stats.critic_continuation_costs_by_action_type["DEL"])
        self.assertEqual([2.], stats.critic_new_totals_by_action_type["DEL"])
        self.assertEqual(vocabulary.COPY, sample.action_history[1].item())

    def test_precompute_critic_model_action_can_augment_costs_not_optimal_mask(self):
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
        copy_action = ConditionalCopy()
        end_action = transducer_.vocab.decode_action(vocabulary.END_WORD)

        def expert_action_scores(input_, target, alignment, output):
            if alignment == 0:
                return {copy_action: 0.}
            return {end_action: 0.}

        transducer_.expert_action_scores = expert_action_scores
        transducer_.expert_score_action = \
            lambda input_, target, alignment, output, action_id: 1000.5

        def expert_score_decoder_action(input_, target, alignment, output, action_id):
            total = 2. if action_id == vocabulary.DELETE else 5.
            return optimal_expert_substitutions.DecoderStateScore(
                prefix_cost=1.,
                continuation_cost=total - 1.,
                total=total,
                target_prefix_index=1,
            )

        transducer_.expert_score_decoder_action = expert_score_decoder_action
        original_rollin_action = train.model_greedy_rollin_action
        model_actions = iter([vocabulary.DELETE, vocabulary.END_WORD])
        train.model_greedy_rollin_action = \
            lambda sample, model, alignments, actions, outputs: next(model_actions)
        sample = utils.Sample(["a"], ["a"], torch.tensor([0, 4, 1]))
        try:
            stats = train.precompute_from_expert(
                sample,
                transducer_,
                device="cpu",
                critic_model_action=True,
                critic_augment_model_action=True,
            )
        finally:
            train.model_greedy_rollin_action = original_rollin_action

        self.assertTrue(sample.optimal_actions_mask[0, vocabulary.COPY])
        self.assertFalse(sample.optimal_actions_mask[0, vocabulary.DELETE])
        self.assertEqual(5., sample.expert_action_costs[0, vocabulary.COPY].item())
        self.assertEqual(2., sample.expert_action_costs[0, vocabulary.DELETE].item())
        self.assertEqual(1, stats.critic_better_actions)
        self.assertEqual([-3.], stats.critic_finite_gaps)

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

    def test_log_focal_statistics_reports_epoch_diagnostics(self):
        stats = {
            "oracle_masses": [
                torch.tensor([0.25, 0.50]),
                torch.tensor([0.75, 1.00]),
            ],
            "focal_weights": [
                torch.tensor([0.75, 0.50]),
                torch.tensor([0.25, 0.00]),
            ],
        }

        with self.assertLogs(level="INFO") as logs:
            train.log_focal_statistics(stats)

        output = "\n".join(logs.output)
        self.assertIn("Focal states: 4.", output)
        self.assertIn("Focal oracle mass: mean 0.6250", output)
        self.assertIn("Focal weight: mean 0.3750", output)

    def test_log_contrastive_statistics_reports_epoch_diagnostics(self):
        stats = {
            "states": 10,
            "active": 4,
            "ranking_correct": 6,
            "margin_satisfied": 3,
            "decoder_top_states": 10,
            "decoder_top_optimal": 6,
            "decoder_top_finite": 7,
            "decoder_top_excluded": 3,
            "loss_sum": 5.,
            "gap_sum": 12.,
            "gap_count": 4,
            "negative_finite_expert": 4,
            "negative_expert_excluded": 6,
            "positive_count_sum": 14,
            "positive_count_max": 3,
            "negative_type_counts": {"DEL": 3, "SUB": 1},
            "decoder_error_type_counts": {"INS": 2, "SUB": 2},
            "delete_gap_sum": 9.,
            "delete_gap_count": 3,
            "delete_logit_advantage_sum": 1.5,
            "delete_ranking_correct": 1,
            "delete_margin_satisfied": 1,
        }

        with self.assertLogs(level="INFO") as logs:
            train.log_contrastive_statistics(stats)

        output = "\n".join(logs.output)
        self.assertIn("Contrastive states: 10.", output)
        self.assertIn("Contrastive accuracy: 0.6000.", output)
        self.assertIn("Decoder top action: optimal 0.6000 finite-expert 0.7000 excluded 0.3000.", output)
        self.assertIn("Contrastive positive set size: mean 1.4000 max 3.", output)
        self.assertIn("finite expert cost 4 (40.00%), no expert cost 6 (60.00%)", output)
        self.assertIn("DEL: 3", output)
        self.assertIn("Decoder top nonoptimal action types:", output)
        self.assertIn("INS: 2", output)
        self.assertIn("DELETE hard negatives: count 3", output)

    def test_log_critic_stats_reports_gap_distribution(self):
        stats = train.RollinStats(
            critic_states=6,
            critic_new_actions=5,
            critic_better_actions=1,
            critic_equal_actions=1,
            critic_regret_sum=18.,
            critic_finite_gaps=[0., 0.5, 1.5, 6., 10.5],
            critic_infinite_gaps=1,
            critic_gaps_by_action_type={
                "SUB": [0.5, 1.5],
                "EOS": [10.5],
            },
            critic_infinite_gaps_by_action_type={
                "INS": 1,
            },
            critic_old_costs_by_action_type={
                "SUB": [1000., 1002.],
            },
            critic_prefix_costs_by_action_type={
                "SUB": [1., 3.],
            },
            critic_continuation_costs_by_action_type={
                "SUB": [4., 6.],
            },
            critic_new_totals_by_action_type={
                "SUB": [5., 9.],
            },
        )

        with self.assertLogs(level="INFO") as logs:
            train.log_critic_stats(stats)

        output = "\n".join(logs.output)
        self.assertIn("Novel critic cost-gap distribution:", output)
        self.assertIn("0 < gap <= 1: 1", output)
        self.assertIn("invalid/infinite: 1", output)
        self.assertIn("SUB count=2 median=1.0000 <=1=1 invalid=0", output)
        self.assertIn("INS count=1 median=nan <=1=0 invalid=1", output)
        self.assertIn("Decoder-state critic costs by action type:", output)
        self.assertIn(
            "SUB count=2 old-median=1001.0000 prefix-median=2.0000 "
            "continuation-median=5.0000 new-total-median=7.0000",
            output,
        )

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

    def test_learning_rates_changed(self):
        self.assertFalse(train.learning_rates_changed([0.1], [0.1]))
        self.assertTrue(train.learning_rates_changed([0.1], [0.05]))

    def test_set_optimizer_learning_rates_updates_all_param_groups(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(
            [
                {"params": [model.weight], "lr": 0.1},
                {"params": [model.bias], "lr": 0.01},
            ],
        )

        train.set_optimizer_learning_rates(optimizer, [0.05, 0.005])

        self.assertEqual([0.05, 0.005], train.optimizer_learning_rates(optimizer))

    def test_set_optimizer_learning_rates_rejects_wrong_length(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaises(ValueError):
            train.set_optimizer_learning_rates(optimizer, [0.1, 0.01])

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

    def test_reload_best_model_and_reset_optimizer_restores_weights_and_clears_state(self):
        vocabularies = self.small_vocabularies()
        args = self.small_transducer_args(device="cpu")
        args.optimizer = "adam"
        args.scheduler = None
        args.lr = 0.001
        args.betas = (0.9, 0.999)
        args.eps = 1e-8
        args.weight_decay = 0.
        args.amsgrad = False
        model = transducer.Transducer(vocabularies, None, args)
        with tempfile.TemporaryDirectory() as tmpdir:
            best_model_path = os.path.join(tmpdir, "best.model")
            torch.save(model.state_dict(), best_model_path)
            saved_first_parameter = next(model.parameters()).detach().clone()
            with torch.no_grad():
                next(model.parameters()).add_(1.)

            optimizer, scheduler = train.reload_best_model_and_reset_optimizer(
                model,
                args,
                best_model_path,
                [0.0005],
            )

        self.assertTrue(torch.allclose(saved_first_parameter, next(model.parameters())))
        self.assertEqual([0.0005], train.optimizer_learning_rates(optimizer))
        self.assertEqual(0, len(optimizer.state))
        self.assertIsNone(scheduler)


if __name__ == "__main__":
    unittest.main()

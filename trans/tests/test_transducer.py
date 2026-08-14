"""Unit tests for transducer.py."""
import unittest
import argparse

import numpy as np
from scipy.special import log_softmax

import torch

from trans import optimal_expert
from trans import transducer
from trans import utils
from trans import vocabulary
from trans.actions import Copy, ConditionalCopy, ConditionalDel, \
    ConditionalIns, ConditionalSub, Sub


np.random.seed(1)


class TransducerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        vocabulary_ = vocabulary.Vocabularies()
        vocabulary_.encode_input("foo")
        vocabulary_.encode_actions("bar")
        expert = optimal_expert.OptimalExpert()

        args = argparse.Namespace(
            device='cpu',
            char_dim=100,
            action_dim=100,
            enc_type='lstm',
            enc_hidden_dim=200,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            dec_hidden_dim=100,
            dec_layers=1
        )
        cls.transducer = transducer.Transducer(
            vocabulary_, expert, args)

    @staticmethod
    def build_small_transducer(output_feedback_dim=0):
        vocabulary_ = vocabulary.Vocabularies(characters=["a"])
        vocabulary_.encode_actions("a")
        expert = optimal_expert.OptimalExpert()

        args = argparse.Namespace(
            device='cpu',
            char_dim=4,
            action_dim=4,
            enc_type='lstm',
            enc_hidden_dim=4,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            enc_output_dropout=0.,
            enc_output_dropout_type="locked",
            output_feedback_dim=output_feedback_dim,
            expert_temperature=0.,
            expert_loss="marginal",
            dec_hidden_dim=4,
            dec_layers=1
        )
        return transducer.Transducer(vocabulary_, expert, args)

    @staticmethod
    def encoded_input(vocabularies, input_):
        return torch.tensor(
            [vocabularies.encode_unseen_input(input_)],
            dtype=torch.long,
        )

    def assert_actions_are_valid_for_input(self, transducer_, input_, action_history):
        alignment = 0
        for action in action_history:
            suffix_length = len(input_) + 1 - alignment
            valid_actions = transducer_.compute_valid_actions(suffix_length)
            self.assertTrue(
                valid_actions[action],
                f"Invalid action {action} at alignment {alignment} "
                f"for input length {len(input_)}.",
            )
            alignment += transducer_.alignment_update[action].item()
            if action == vocabulary.END_WORD:
                break

    def test_sample(self):
        log_probs = log_softmax([5, 4, 10, 1])
        action_code = self.transducer.sample(log_probs)
        self.assertTrue(0 <= action_code < self.transducer.number_actions)

    def test_compute_valid_actions(self):
        valid_actions = self.transducer.compute_valid_actions(3)
        self.assertTrue(self.transducer.number_actions, len(valid_actions))
        valid_actions = self.transducer.compute_valid_actions(1)
        self.assertTrue(not valid_actions[vocabulary.COPY])

    def test_compute_valid_actions_uses_requested_device(self):
        valid_actions = self.transducer.compute_valid_actions(3, device="cpu")

        self.assertEqual(torch.device("cpu"), valid_actions.device)

    def test_valid_actions_for_suffixes_matches_two_validity_states(self):
        suffix_lengths = torch.tensor([-3, -1, 0, 1, 2, 3, 20])

        valid_actions = self.transducer.valid_actions_for_suffixes(suffix_lengths)

        self.assertEqual(
            (1, len(suffix_lengths), self.transducer.number_actions),
            tuple(valid_actions.shape),
        )
        self.assertEqual(torch.bool, valid_actions.dtype)
        for i, suffix_length in enumerate(suffix_lengths.tolist()):
            expected = self.transducer.compute_valid_actions(max(suffix_length, 0))
            self.assertTrue(torch.equal(expected, valid_actions[0, i]))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_valid_actions_for_suffixes_accepts_mps_lengths(self):
        suffix_lengths = torch.tensor([0, 1, 100], device="mps")

        valid_actions = self.transducer.valid_actions_for_suffixes(suffix_lengths)

        self.assertEqual(torch.device("cpu"), valid_actions.device)

    def test_encode_known_action_rejects_action_absent_from_vocabulary(self):
        transducer_ = self.build_small_transducer()

        with self.assertRaisesRegex(RuntimeError, "absent from action vocabulary"):
            transducer_.encode_known_action(ConditionalIns("z"), "test")

    def test_encode_expert_action_costs_uses_requested_device(self):
        transducer_ = self.build_small_transducer()

        costs = transducer_.encode_expert_action_costs(
            {ConditionalCopy(): 0.},
            device="cpu",
        )

        self.assertEqual(torch.device("cpu"), costs.device)

    def test_validate_index_tensor_rejects_out_of_range_values(self):
        transducer_ = self.build_small_transducer()

        with self.assertRaisesRegex(RuntimeError, "test ids contains ids outside"):
            transducer_.validate_index_tensor(
                "test ids",
                torch.tensor([0, transducer_.number_actions]),
                transducer_.number_actions,
            )

    def test_decode_encoded_output_uses_target_separator(self):
        transducer_ = self.build_small_transducer()
        transducer_.target_tokenizer = utils.Tokenizer(" ")
        transducer_.vocab.encode_actions(["d͡ʒ"])
        phone_id = transducer_.vocab.encode_unseen_action(ConditionalIns("d͡ʒ"))

        decoded_output = transducer_.decode_encoded_output(
            [["a"]],
            [[phone_id, vocabulary.COPY]],
        )

        self.assertEqual(["d͡ʒ a"], decoded_output)

    def test_output_symbol_for_action(self):
        transducer_ = self.build_small_transducer()
        transducer_.vocab.encode_actions(["x"])
        source = ["a", "b"]

        self.assertEqual("a", transducer_.output_symbol_for_action(source, vocabulary.COPY, 0))
        self.assertEqual("b", transducer_.output_symbol_for_action(source, vocabulary.COPY, 1))
        self.assertEqual("x", transducer_.output_symbol_for_action(source, ConditionalSub("x"), 1))
        self.assertEqual("x", transducer_.output_symbol_for_action(source, ConditionalIns("x"), 0))
        self.assertEqual(
            vocabulary.NO_OUTPUT,
            transducer_.output_symbol_for_action(source, vocabulary.DELETE, 0),
        )
        self.assertEqual(
            vocabulary.BOS_OUTPUT,
            transducer_.output_symbol_for_action(source, vocabulary.BEGIN_WORD, 0),
        )

    def test_output_feedback_disabled_has_legacy_decoder_input_dim(self):
        transducer_ = self.build_small_transducer(output_feedback_dim=0)

        self.assertIsNone(transducer_.output_lookup)
        self.assertEqual(
            transducer_.enc.output_size + 4,
            transducer_.dec.input_size,
        )

    def test_output_feedback_extends_decoder_input_dim(self):
        transducer_ = self.build_small_transducer(output_feedback_dim=3)

        self.assertIsNotNone(transducer_.output_lookup)
        self.assertEqual(
            transducer_.enc.output_size + 4 + 3,
            transducer_.dec.input_size,
        )

    def test_soft_oracle_loss_uses_cost_gaps(self):
        transducer_ = self.build_small_transducer()
        logits = torch.log(torch.tensor([[[0.6, 0.4, 0.0]]]))
        valid_actions = torch.tensor([[[True, True, False]]])
        expert_costs = torch.tensor([[[1.0, 1.2, float("inf")]]])

        log_mass = transducer_.soft_oracle_loss(
            logits,
            expert_costs,
            valid_actions,
            temperature=0.2,
        )

        expected = np.log(0.6 + 0.4 * np.exp(-1.0))
        self.assertTrue(torch.isclose(log_mass[0, 0], torch.tensor(expected, dtype=torch.float)))

    def test_soft_oracle_loss_handles_padded_timesteps(self):
        transducer_ = self.build_small_transducer()
        logits = torch.randn(2, 1, transducer_.number_actions, requires_grad=True)
        valid_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        valid_actions[0, 0, [vocabulary.END_WORD, vocabulary.COPY]] = True
        expert_costs = torch.full((2, 1, transducer_.number_actions), float("inf"))
        expert_costs[0, 0, vocabulary.END_WORD] = 0.
        expert_costs[0, 0, vocabulary.COPY] = 1.

        log_mass = transducer_.soft_oracle_loss(
            logits,
            expert_costs,
            valid_actions,
            temperature=4.,
        )
        loss = -log_mass.sum()
        loss.backward()

        self.assertTrue(torch.isfinite(log_mass).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(torch.zeros_like(logits.grad[1]), logits.grad[1]))

    def test_focal_marginal_loss_gamma_zero_matches_marginal_loss_and_gradient(self):
        transducer_ = self.build_small_transducer()
        logits = torch.randn(
            2,
            1,
            transducer_.number_actions,
            requires_grad=True,
        )
        valid_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        valid_actions[0, 0, [vocabulary.END_WORD, vocabulary.COPY]] = True
        valid_actions[1, 0, [vocabulary.END_WORD, vocabulary.DELETE]] = True
        optimal_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        optimal_actions[0, 0, vocabulary.COPY] = True
        optimal_actions[1, 0, vocabulary.END_WORD] = True

        marginal = -transducer_.log_sum_softmax_loss(
            logits,
            optimal_actions,
            valid_actions,
        )
        focal = transducer_.focal_marginal_loss(
            logits,
            optimal_actions,
            valid_actions,
            gamma=0.,
        )
        marginal.sum().backward(retain_graph=True)
        marginal_grad = logits.grad.clone()
        logits.grad.zero_()
        focal.sum().backward()
        focal_grad = logits.grad.clone()

        self.assertTrue(torch.allclose(marginal, focal, atol=1e-6))
        self.assertTrue(torch.allclose(marginal_grad, focal_grad, atol=1e-6))

    def test_focal_marginal_loss_downweights_easy_states(self):
        transducer_ = self.build_small_transducer()
        logits = torch.log(torch.tensor([[[0.9, 0.1, 0.0]]]))
        valid_actions = torch.tensor([[[True, True, False]]])
        optimal_actions = torch.tensor([[[True, False, False]]])

        focal = transducer_.focal_marginal_loss(
            logits,
            optimal_actions,
            valid_actions,
            gamma=2.,
        )

        expected = -((1. - 0.9) ** 2) * np.log(0.9)
        self.assertTrue(torch.isclose(
            focal[0, 0],
            torch.tensor(expected, dtype=torch.float),
        ))

    def test_focal_marginal_loss_handles_padded_timesteps(self):
        transducer_ = self.build_small_transducer()
        logits = torch.randn(2, 1, transducer_.number_actions, requires_grad=True)
        valid_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        valid_actions[0, 0, [vocabulary.END_WORD, vocabulary.COPY]] = True
        optimal_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        optimal_actions[0, 0, vocabulary.END_WORD] = True

        losses = transducer_.focal_marginal_loss(
            logits,
            optimal_actions,
            valid_actions,
            gamma=1.,
        )
        losses.sum().backward()

        self.assertTrue(torch.isfinite(losses).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(torch.zeros_like(logits.grad[1]), logits.grad[1]))

    def test_normalized_soft_expert_loss_matches_expected_cross_entropy(self):
        transducer_ = self.build_small_transducer()
        logits = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]]))
        valid_actions = torch.tensor([[[True, True, True]]])
        expert_costs = torch.tensor([[[1.0, 1.2, float("inf")]]])

        log_mass = transducer_.normalized_soft_expert_loss(
            logits,
            expert_costs,
            valid_actions,
            temperature=0.2,
        )

        expert_weights = torch.tensor([1.0, np.exp(-1.0)])
        expert_probs = expert_weights / expert_weights.sum()
        expected = (
            expert_probs[0] * torch.log(torch.tensor(0.6)) +
            expert_probs[1] * torch.log(torch.tensor(0.3))
        ).float()
        self.assertTrue(torch.isclose(log_mass[0, 0], expected))

    def test_normalized_soft_expert_loss_handles_padded_timesteps(self):
        transducer_ = self.build_small_transducer()
        logits = torch.randn(2, 1, transducer_.number_actions, requires_grad=True)
        valid_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        valid_actions[0, 0, [vocabulary.END_WORD, vocabulary.COPY]] = True
        expert_costs = torch.full((2, 1, transducer_.number_actions), float("inf"))
        expert_costs[0, 0, vocabulary.END_WORD] = 0.
        expert_costs[0, 0, vocabulary.COPY] = 1.

        log_mass = transducer_.normalized_soft_expert_loss(
            logits,
            expert_costs,
            valid_actions,
            temperature=4.,
        )
        loss = -log_mass.sum()
        loss.backward()

        self.assertTrue(torch.isfinite(log_mass).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(torch.zeros_like(logits.grad[1]), logits.grad[1]))

    def test_margin_expert_loss_uses_best_oracle_and_best_decoder_valid_competitor(self):
        transducer_ = self.build_small_transducer()
        logits = torch.tensor([[[0.0, 2.0, 1.5, 4.0]]])
        valid_actions = torch.tensor([[[True, True, True, True]]])
        optimal_actions = torch.tensor([[[False, True, True, False]]])

        losses = transducer_.margin_expert_loss(
            logits,
            optimal_actions,
            valid_actions,
            margin=1.0,
        )

        self.assertTrue(torch.equal(torch.tensor([[3.0]]), losses))
        self.assertEqual(1, transducer_.last_margin_statistics["states"])
        self.assertEqual(1, transducer_.last_margin_statistics["active"])
        self.assertEqual(-2.0, transducer_.last_margin_statistics["margin_sum"])
        self.assertEqual(3.0, transducer_.last_margin_statistics["active_loss_sum"])

    def test_margin_expert_loss_is_zero_after_margin_is_satisfied(self):
        transducer_ = self.build_small_transducer()
        logits = torch.tensor([[[0.0, 5.0, 1.0]]])
        valid_actions = torch.tensor([[[True, True, True]]])
        optimal_actions = torch.tensor([[[False, True, False]]])

        losses = transducer_.margin_expert_loss(
            logits,
            optimal_actions,
            valid_actions,
            margin=1.0,
        )

        self.assertTrue(torch.equal(torch.tensor([[0.0]]), losses))
        self.assertEqual(0, transducer_.last_margin_statistics["active"])

    def test_margin_expert_loss_handles_padded_timesteps(self):
        transducer_ = self.build_small_transducer()
        logits = torch.randn(2, 1, transducer_.number_actions, requires_grad=True)
        valid_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        valid_actions[0, 0, [vocabulary.END_WORD, vocabulary.COPY]] = True
        optimal_actions = torch.zeros(2, 1, transducer_.number_actions, dtype=torch.bool)
        optimal_actions[0, 0, vocabulary.END_WORD] = True

        losses = transducer_.margin_expert_loss(
            logits,
            optimal_actions,
            valid_actions,
            margin=1.0,
        )
        loss = losses.sum()
        loss.backward()

        self.assertTrue(torch.isfinite(losses).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(torch.zeros_like(logits.grad[1]), logits.grad[1]))

    def test_decoder_step_clamps_alignment_lookup_indices(self):
        transducer_ = self.build_small_transducer()
        transducer_.h0_c0 = 1
        encoder_output = torch.zeros(2, 1, transducer_.enc.output_size)
        alignment = torch.tensor([-1, 0, 5])
        action_history = torch.tensor([
            [vocabulary.BEGIN_WORD],
            [vocabulary.COPY],
            [vocabulary.END_WORD],
        ])

        decoder_output, _ = transducer_.decoder_step(
            encoder_output,
            feature_embedding=None,
            decoder_cell_state=transducer_.h0_c0,
            alignment=alignment,
            action_history=action_history,
        )

        self.assertEqual((3, 1, transducer_.dec_hidden_dim), tuple(decoder_output.shape))

    def test_decoder_step_gathers_encoder_outputs_by_alignment(self):
        transducer_ = self.build_small_transducer()
        transducer_.h0_c0 = 2
        seen_decoder_input = {}
        encoder_output = torch.arange(
            3 * 2 * transducer_.enc.output_size,
            dtype=torch.float,
        ).view(3, 2, transducer_.enc.output_size)
        alignment = torch.tensor([0, 2, 1, 0])
        action_history = torch.tensor([
            [vocabulary.BEGIN_WORD, vocabulary.BEGIN_WORD],
            [vocabulary.COPY, vocabulary.END_WORD],
        ])

        class Decoder(torch.nn.Module):
            def forward(self, decoder_input, decoder_cell_state):
                seen_decoder_input["value"] = decoder_input.detach()
                return (
                    torch.zeros(
                        decoder_input.size(0),
                        decoder_input.size(1),
                        transducer_.dec_hidden_dim,
                    ),
                    decoder_cell_state,
                )

        transducer_.dec = Decoder()

        transducer_.decoder_step(
            encoder_output,
            feature_embedding=None,
            decoder_cell_state=transducer_.h0_c0,
            alignment=alignment,
            action_history=action_history,
        )

        expected_encoder_part = torch.stack([
            encoder_output[0, 0],
            encoder_output[1, 1],
            encoder_output[2, 0],
            encoder_output[0, 1],
        ]).view(2, 2, -1)
        actual_encoder_part = seen_decoder_input["value"][:, :, :transducer_.enc.output_size]
        self.assertTrue(torch.equal(expected_encoder_part, actual_encoder_part))

    def test_decoder_step_maps_negative_padded_history_ids_to_pad(self):
        transducer_ = self.build_small_transducer(output_feedback_dim=2)
        transducer_.h0_c0 = 1
        encoder_output = torch.zeros(2, 1, transducer_.enc.output_size)
        alignment = torch.tensor([0, 0, 0])
        action_history = torch.tensor([
            [vocabulary.BEGIN_WORD],
            [-1],
            [vocabulary.END_WORD],
        ])
        output_history = torch.tensor([
            [transducer_.vocab.encode_output_symbol(vocabulary.BOS_OUTPUT)],
            [-1],
            [transducer_.vocab.encode_output_symbol(vocabulary.NO_OUTPUT)],
        ])

        decoder_output, _ = transducer_.decoder_step(
            encoder_output,
            feature_embedding=None,
            decoder_cell_state=transducer_.h0_c0,
            alignment=alignment,
            action_history=action_history,
            output_history=output_history,
        )

        self.assertEqual((3, 1, transducer_.dec_hidden_dim), tuple(decoder_output.shape))

    def test_training_step_clears_stale_margin_statistics_for_non_margin_loss(self):
        transducer_ = self.build_small_transducer()
        transducer_.last_margin_statistics = {"states": 1}
        encoded_input = self.encoded_input(transducer_.vocab, ["a"])
        action_history = torch.tensor([[vocabulary.BEGIN_WORD]])
        alignment_history = torch.tensor([0])
        optimal_actions = torch.zeros(1, 1, transducer_.number_actions, dtype=torch.bool)
        optimal_actions[0, 0, vocabulary.COPY] = True
        valid_actions = torch.ones(1, 1, transducer_.number_actions, dtype=torch.bool)

        def encoder_step(encoded_input, is_training=False):
            return torch.zeros(1, 1, transducer_.enc.output_size)

        def decoder_step(encoder_output, feature_embedding, decoder_cell_state,
                         alignment, action_history, output_history=None):
            return (
                torch.zeros(1, 1, transducer_.dec_hidden_dim),
                decoder_cell_state,
            )

        transducer_.encoder_step = encoder_step
        transducer_.decoder_step = decoder_step

        transducer_.training_step(
            encoded_input=encoded_input,
            encoded_features=None,
            action_history=action_history,
            output_history=None,
            alignment_history=alignment_history,
            expert_action_costs=None,
            optimal_actions_mask=optimal_actions,
            valid_actions_mask=valid_actions,
        )

        self.assertIsNone(transducer_.last_margin_statistics)

    def test_transduce_passes_previous_output_feedback(self):
        transducer_ = self.build_small_transducer(output_feedback_dim=2)
        action_script = [vocabulary.COPY, vocabulary.END_WORD]
        seen_output_history = []
        step = {"i": 0}

        def encoder_step(encoded_input, is_training=False):
            return torch.zeros(
                encoded_input.size(1) - 1,
                encoded_input.size(0),
                transducer_.enc.output_size,
            )

        def decoder_step(encoder_output, feature_embedding, decoder_cell_state,
                         alignment, action_history, output_history=None):
            seen_output_history.append(output_history.clone())
            return (
                torch.zeros(1, encoder_output.size(1), transducer_.dec_hidden_dim),
                decoder_cell_state,
            )

        def calculate_actions(decoder_output, valid_actions_mask):
            action = action_script[step["i"]]
            step["i"] += 1
            actions = torch.tensor([[action]], dtype=torch.long)
            log_probs = torch.full((1, 1, transducer_.number_actions), -1000.0)
            log_probs[0, 0, action] = -0.1
            return actions, log_probs

        transducer_.encoder_step = encoder_step
        transducer_.decoder_step = decoder_step
        transducer_.calculate_actions = calculate_actions

        transducer_.transduce(
            [["a"]],
            self.encoded_input(transducer_.vocab, ["a"]),
            encoded_features=None,
        )

        expected = [
            transducer_.vocab.encode_output_symbol(vocabulary.BOS_OUTPUT),
            transducer_.vocab.encode_output_symbol("a"),
        ]
        actual = [history.item() for history in seen_output_history]
        self.assertEqual(expected, actual)

    def test_log_sum_softmax_loss_ignores_invalid_optimal_actions(self):
        logits = torch.tensor([[[0., 10., 2.]]])
        valid_actions_mask = torch.tensor([[[True, False, True]]])
        optimal_actions_mask = torch.tensor([[[False, True, True]]])

        loss = self.transducer.log_sum_softmax_loss(
            logits, optimal_actions_mask, valid_actions_mask)

        expected = torch.tensor([[2.]]) - torch.logsumexp(
            torch.tensor([[0., 2.]]), dim=1)
        self.assertTrue(torch.allclose(expected, loss))

    def test_greedy_decode_accepts_boundary_input_lengths(self):
        transducer_ = self.build_small_transducer()

        for length in (0, 1, 99, 100, 250):
            with self.subTest(length=length):
                input_ = "a" * length
                output = transducer_.transduce(
                    [input_],
                    self.encoded_input(transducer_.vocab, input_),
                    encoded_features=None,
                )
                self.assertEqual(1, len(output.action_history))
                self.assert_actions_are_valid_for_input(
                    transducer_, input_, output.action_history[0])

    def test_greedy_decode_accepts_mixed_length_batches(self):
        transducer_ = self.build_small_transducer()
        inputs = ["", "a", "a" * 100, "a" * 250]
        encoded_inputs = [
            torch.tensor(transducer_.vocab.encode_unseen_input(input_),
                         dtype=torch.long)
            for input_ in inputs
        ]

        output = transducer_.transduce(
            inputs,
            torch.nn.utils.rnn.pad_sequence(
                encoded_inputs,
                batch_first=True,
                padding_value=vocabulary.PAD,
            ),
            encoded_features=None,
        )

        self.assertEqual(len(inputs), len(output.action_history))
        for input_, action_history in zip(inputs, output.action_history):
            self.assert_actions_are_valid_for_input(
                transducer_, input_, action_history)

    def test_beam_decode_accepts_boundary_input_lengths(self):
        transducer_ = self.build_small_transducer()

        for length in (0, 1, 99, 100, 250):
            with self.subTest(length=length):
                input_ = "a" * length
                outputs = transducer_.beam_search_decode(
                    input_,
                    self.encoded_input(transducer_.vocab, input_),
                    encoded_features=None,
                    beam_width=2,
                )
                self.assertGreaterEqual(len(outputs), 1)
                for output in outputs:
                    self.assert_actions_are_valid_for_input(
                        transducer_, input_, output.action_history)

    def test_beam_decode_keeps_live_beam_width_after_early_eos(self):
        transducer_ = self.build_small_transducer()
        decoder_calls = {"count": 0}

        def encoder_step(encoded_input, is_training=False):
            return torch.zeros(1, 1, transducer_.enc.output_size)

        def decoder_step(encoder_output, feature_embedding, decoder_cell_state,
                         alignment, action_history, output_history=None):
            decoder_calls["count"] += 1
            return (
                torch.zeros(1, 1, transducer_.dec_hidden_dim),
                decoder_cell_state,
            )

        def log_softmax_(logits, valid_actions_mask):
            action_len = valid_actions_mask.size(2)
            log_probs = torch.full((1, 1, action_len), -1000.0)
            previous_action = int(current_action_history["value"][-1].item())
            if previous_action == vocabulary.BEGIN_WORD:
                log_probs[0, 0, vocabulary.END_WORD] = -0.1
                log_probs[0, 0, vocabulary.COPY] = -0.2
            elif previous_action == vocabulary.COPY:
                log_probs[0, 0, vocabulary.END_WORD] = -0.1
            return log_probs

        current_action_history = {"value": None}

        def decoder_step_with_history(encoder_output, feature_embedding,
                                      decoder_cell_state, alignment,
                                      action_history, output_history=None):
            current_action_history["value"] = action_history
            return decoder_step(
                encoder_output,
                feature_embedding,
                decoder_cell_state,
                alignment,
                action_history,
                output_history,
            )

        transducer_.encoder_step = encoder_step
        transducer_.decoder_step = decoder_step_with_history
        transducer_.log_softmax = log_softmax_

        outputs = transducer_.beam_search_decode(
            ["a"],
            self.encoded_input(transducer_.vocab, ["a"]),
            encoded_features=None,
            beam_width=2,
        )

        self.assertEqual(2, len(outputs))
        self.assertGreaterEqual(decoder_calls["count"], 2)
        self.assertEqual([vocabulary.END_WORD], outputs[0].action_history)
        self.assertEqual([vocabulary.COPY, vocabulary.END_WORD], outputs[1].action_history)

    def test_beam_decode_logs_debug_stats(self):
        transducer_ = self.build_small_transducer()

        with self.assertLogs(level="DEBUG") as logs:
            transducer_.beam_search_decode(
                ["a"],
                self.encoded_input(transducer_.vocab, ["a"]),
                encoded_features=None,
                beam_width=1,
            )

        self.assertIn("Beam stats: requested_width=1", "\n".join(logs.output))

    def test_transduce_finished_sequences_become_inert(self):
        transducer_ = self.build_small_transducer()
        action_script = [
            [vocabulary.END_WORD, vocabulary.COPY],
            [vocabulary.COPY, vocabulary.COPY],
            [vocabulary.COPY, vocabulary.END_WORD],
        ]
        logp_script = [
            [-0.5, -1.0],
            [-100.0, -2.0],
            [-100.0, -3.0],
        ]
        alignments = []
        step = {"i": 0}

        def encoder_step(encoded_input, is_training=False):
            return torch.zeros(
                encoded_input.size(1) - 1,
                encoded_input.size(0),
                transducer_.enc.output_size,
            )

        def decoder_step(encoder_output, feature_embedding, decoder_cell_state,
                         alignment, action_history, output_history=None):
            alignments.append(alignment.clone().cpu().tolist())
            return (
                torch.zeros(1, encoder_output.size(1), transducer_.dec_hidden_dim),
                decoder_cell_state,
            )

        def calculate_actions(decoder_output, valid_actions_mask):
            action_ids = torch.tensor(
                [action_script[step["i"]]],
                dtype=torch.long,
            )
            log_probs = torch.full(
                (1, 2, transducer_.number_actions),
                -1000.0,
            )
            for batch_index, action_id in enumerate(action_script[step["i"]]):
                log_probs[0, batch_index, action_id] = logp_script[step["i"]][batch_index]
            step["i"] += 1
            return action_ids, log_probs

        transducer_.encoder_step = encoder_step
        transducer_.decoder_step = decoder_step
        transducer_.calculate_actions = calculate_actions

        inputs = ["a", "aa"]
        encoded_inputs = [
            torch.tensor(transducer_.vocab.encode_unseen_input(input_),
                         dtype=torch.long)
            for input_ in inputs
        ]
        output = transducer_.transduce(
            inputs,
            torch.nn.utils.rnn.pad_sequence(
                encoded_inputs,
                batch_first=True,
                padding_value=vocabulary.PAD,
            ),
            encoded_features=None,
        )

        self.assertEqual([[vocabulary.END_WORD], [vocabulary.COPY, vocabulary.COPY, vocabulary.END_WORD]],
                         output.action_history)
        self.assertEqual(["", "aa"], output.output)
        self.assertTrue(np.isclose(-1.25, output.log_p))
        self.assertEqual([[0, 0], [0, 1], [0, 2]], alignments)

    def test_transduce_batch_invariance_for_early_finished_sequence(self):
        alone = self.build_small_transducer()
        batched = self.build_small_transducer()

        def patch(transducer_, action_script, logp_script):
            step = {"i": 0}

            def encoder_step(encoded_input, is_training=False):
                return torch.zeros(
                    encoded_input.size(1) - 1,
                    encoded_input.size(0),
                    transducer_.enc.output_size,
                )

            def decoder_step(encoder_output, feature_embedding,
                             decoder_cell_state, alignment, action_history,
                             output_history=None):
                return (
                    torch.zeros(1, encoder_output.size(1), transducer_.dec_hidden_dim),
                    decoder_cell_state,
                )

            def calculate_actions(decoder_output, valid_actions_mask):
                actions = torch.tensor([action_script[step["i"]]], dtype=torch.long)
                log_probs = torch.full(
                    (1, len(action_script[step["i"]]), transducer_.number_actions),
                    -1000.0,
                )
                for batch_index, action_id in enumerate(action_script[step["i"]]):
                    log_probs[0, batch_index, action_id] = logp_script[step["i"]][batch_index]
                step["i"] += 1
                return actions, log_probs

            transducer_.encoder_step = encoder_step
            transducer_.decoder_step = decoder_step
            transducer_.calculate_actions = calculate_actions

        patch(alone, [[vocabulary.END_WORD]], [[-0.5]])
        single = alone.transduce(
            ["a"],
            self.encoded_input(alone.vocab, "a"),
            encoded_features=None,
        )

        patch(
            batched,
            [
                [vocabulary.END_WORD, vocabulary.COPY],
                [vocabulary.COPY, vocabulary.COPY],
                [vocabulary.COPY, vocabulary.END_WORD],
            ],
            [
                [-0.5, -0.5],
                [-100.0, -0.5],
                [-100.0, -0.5],
            ],
        )
        encoded_inputs = [
            torch.tensor(batched.vocab.encode_unseen_input(input_), dtype=torch.long)
            for input_ in ["a", "aa"]
        ]
        batch = batched.transduce(
            ["a", "aa"],
            torch.nn.utils.rnn.pad_sequence(
                encoded_inputs,
                batch_first=True,
                padding_value=vocabulary.PAD,
            ),
            encoded_features=None,
        )

        self.assertEqual(single.action_history[0], batch.action_history[0])
        self.assertEqual(single.output[0], batch.output[0])
        self.assertTrue(np.isclose(single.log_p, batch.log_p))

    def test_transduce_all_examples_terminate_same_step(self):
        transducer_ = self.build_small_transducer()

        def encoder_step(encoded_input, is_training=False):
            return torch.zeros(
                encoded_input.size(1) - 1,
                encoded_input.size(0),
                transducer_.enc.output_size,
            )

        def decoder_step(encoder_output, feature_embedding, decoder_cell_state,
                         alignment, action_history, output_history=None):
            return (
                torch.zeros(1, encoder_output.size(1), transducer_.dec_hidden_dim),
                decoder_cell_state,
            )

        def calculate_actions(decoder_output, valid_actions_mask):
            actions = torch.tensor([[vocabulary.END_WORD, vocabulary.END_WORD]],
                                   dtype=torch.long)
            log_probs = torch.full((1, 2, transducer_.number_actions), -1000.0)
            log_probs[0, 0, vocabulary.END_WORD] = -0.25
            log_probs[0, 1, vocabulary.END_WORD] = -0.75
            return actions, log_probs

        transducer_.encoder_step = encoder_step
        transducer_.decoder_step = decoder_step
        transducer_.calculate_actions = calculate_actions

        encoded_inputs = [
            torch.tensor(transducer_.vocab.encode_unseen_input(input_), dtype=torch.long)
            for input_ in ["a", "aa"]
        ]
        output = transducer_.transduce(
            ["a", "aa"],
            torch.nn.utils.rnn.pad_sequence(
                encoded_inputs,
                batch_first=True,
                padding_value=vocabulary.PAD,
            ),
            encoded_features=None,
        )

        self.assertEqual([[vocabulary.END_WORD], [vocabulary.END_WORD]],
                         output.action_history)
        self.assertEqual(["", ""], output.output)
        self.assertTrue(np.isclose(-0.5, output.log_p))

    def test_transduce_without_end_word_uses_generated_action_denominator(self):
        transducer_ = self.build_small_transducer()
        action = transducer_.inserts[0]
        step = {"i": 0}
        max_steps = transducer.MAX_ACTION_SEQ_LEN

        def encoder_step(encoded_input, is_training=False):
            return torch.zeros(
                encoded_input.size(1) - 1,
                encoded_input.size(0),
                transducer_.enc.output_size,
            )

        def decoder_step(encoder_output, feature_embedding, decoder_cell_state,
                         alignment, action_history, output_history=None):
            return (
                torch.zeros(1, encoder_output.size(1), transducer_.dec_hidden_dim),
                decoder_cell_state,
            )

        def calculate_actions(decoder_output, valid_actions_mask):
            step["i"] += 1
            actions = torch.tensor([[action]], dtype=torch.long)
            log_probs = torch.full((1, 1, transducer_.number_actions), -1000.0)
            log_probs[0, 0, action] = -1.0
            return actions, log_probs

        transducer_.encoder_step = encoder_step
        transducer_.decoder_step = decoder_step
        transducer_.calculate_actions = calculate_actions

        output = transducer_.transduce(
            ["a"],
            self.encoded_input(transducer_.vocab, "a"),
            encoded_features=None,
        )

        self.assertEqual(max_steps, step["i"])
        self.assertEqual(max_steps, len(output.action_history[0]))
        self.assertEqual(action, output.action_history[0][-1])
        self.assertTrue(np.isclose(-1.0, output.log_p))

    def test_encoded_action_history_trims_at_end_word(self):
        encoded_history = torch.tensor([[
            [vocabulary.BEGIN_WORD, 10, vocabulary.END_WORD, 11],
            [vocabulary.BEGIN_WORD, 12, 13, vocabulary.END_WORD],
        ]])

        action_history = transducer.Transducer.trim_encoded_action_history(
            encoded_history)

        self.assertEqual(
            [[10, vocabulary.END_WORD], [12, 13, vocabulary.END_WORD]],
            action_history,
        )

    def test_encoded_action_history_preserves_final_action_without_end_word(self):
        encoded_history = torch.tensor([[
            [vocabulary.BEGIN_WORD, 10, 11, 12],
        ]])

        action_history = transducer.Transducer.trim_encoded_action_history(
            encoded_history)

        self.assertEqual([[10, 11, 12]], action_history)

    def test_remap_actions(self):
        action_scores = {Copy("w", "w"): 7., Sub("w", "v"): 5.}
        expected = {ConditionalCopy(): 7., ConditionalSub("v"): 5.}
        remapped = self.transducer.remap_actions(action_scores)
        self.assertDictEqual(expected, remapped)

    def test_expert_rollout(self):
        optimal_actions = self.transducer.expert_rollout(
            input_="foo", target="bar", alignment=1, prediction=["b", "a"])
        expected = {self.transducer.vocab.encode_unseen_action(a)
                    for a in (ConditionalIns("r"), ConditionalDel())}
        self.assertSetEqual(expected, set(optimal_actions))


if __name__ == "__main__":
    TransducerTests().run()

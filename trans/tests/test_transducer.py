"""Unit tests for transducer.py."""
import unittest
import argparse

import numpy as np
from scipy.special import log_softmax

import torch

from trans import optimal_expert
from trans import transducer
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
            dec_hidden_dim=100,
            dec_layers=1
        )
        cls.transducer = transducer.Transducer(
            vocabulary_, expert, args)

    @staticmethod
    def build_small_transducer():
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

    def test_valid_actions_for_suffixes_is_dynamic(self):
        suffix_lengths = torch.tensor([0, 1, 99, 100, 250])

        valid_actions = self.transducer.valid_actions_for_suffixes(suffix_lengths)

        self.assertEqual(
            (1, len(suffix_lengths), self.transducer.number_actions),
            tuple(valid_actions.shape),
        )
        self.assertFalse(valid_actions[0, 0, vocabulary.COPY])
        self.assertFalse(valid_actions[0, 1, vocabulary.COPY])
        self.assertTrue(valid_actions[0, 2, vocabulary.COPY])
        self.assertTrue(valid_actions[0, 3, vocabulary.COPY])
        self.assertTrue(valid_actions[0, 4, vocabulary.COPY])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_valid_actions_for_suffixes_accepts_mps_lengths(self):
        suffix_lengths = torch.tensor([0, 1, 100], device="mps")

        valid_actions = self.transducer.valid_actions_for_suffixes(suffix_lengths)

        self.assertEqual(torch.device("cpu"), valid_actions.device)

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

    def test_encoded_action_history_trims_at_end_word(self):
        encoded_history = torch.tensor([[
            [vocabulary.BEGIN_WORD, 10, vocabulary.END_WORD, 11],
            [vocabulary.BEGIN_WORD, 12, 13, vocabulary.END_WORD],
        ]])

        action_history = [
            seq[1:(seq.index(vocabulary.END_WORD) + 1 if vocabulary.END_WORD in seq else -1)]
            for seq in encoded_history.squeeze(dim=0).cpu().tolist()
        ]

        self.assertEqual(
            [[10, vocabulary.END_WORD], [12, 13, vocabulary.END_WORD]],
            action_history,
        )

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

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

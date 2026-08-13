"""Unit tests for encoder modules."""
import argparse
import unittest
import warnings

import torch

from trans import encoders


class TestSinusoidalPositionalEmbedding(unittest.TestCase):

    def test_positions_are_independent_across_batch_rows(self):
        padding_idx = 1
        embedding = encoders.SinusoidalPositionalEmbedding(
            embedding_dim=6,
            padding_idx=padding_idx,
            init_size=4,
        )
        tokens = torch.tensor([
            [2, 3, 4, padding_idx],
            [5, padding_idx, padding_idx, padding_idx],
        ])

        positions = embedding(tokens)

        self.assertTrue(torch.equal(positions[0, 0], positions[1, 0]))
        self.assertTrue(torch.equal(positions[1, 1], torch.zeros(6)))
        self.assertTrue(torch.equal(positions[0, 3], torch.zeros(6)))

    def test_odd_embedding_dimension_and_expansion(self):
        padding_idx = 1
        embedding = encoders.SinusoidalPositionalEmbedding(
            embedding_dim=5,
            padding_idx=padding_idx,
            init_size=2,
        )
        tokens = torch.tensor([[2, 3, 4, 5]])

        positions = embedding(tokens)

        self.assertEqual((1, 4, 5), tuple(positions.shape))
        self.assertGreaterEqual(embedding.weights.size(0), padding_idx + 1 + tokens.size(1))


def lstm_args(**overrides):
    args = argparse.Namespace(
        device="cpu",
        char_dim=4,
        enc_hidden_dim=3,
        enc_layers=1,
        enc_bidirectional=True,
        enc_dropout=0.,
        enc_output_dropout=0.,
        enc_output_dropout_type="locked",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class LSTMOutputDropoutTests(unittest.TestCase):

    @staticmethod
    def sample_input():
        return torch.randn(5, 2, 4)

    def test_zero_output_dropout_matches_legacy_output(self):
        torch.manual_seed(1)
        legacy_encoder = encoders.LSTMEncoder(
            lstm_args(enc_output_dropout=0., enc_output_dropout_type="none"))
        torch.manual_seed(1)
        dropout_encoder = encoders.LSTMEncoder(
            lstm_args(enc_output_dropout=0., enc_output_dropout_type="locked"))
        input_ = self.sample_input()

        legacy_encoder.train()
        dropout_encoder.train()
        legacy_output, _ = legacy_encoder(input_)
        dropout_output, _ = dropout_encoder(input_)

        self.assertTrue(torch.equal(legacy_output, dropout_output))

    def test_single_layer_enc_dropout_alone_does_not_change_training_outputs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder = encoders.LSTMEncoder(
                lstm_args(enc_dropout=0.5, enc_output_dropout=0.))
        input_ = self.sample_input()

        encoder.train()
        first_output, _ = encoder(input_)
        second_output, _ = encoder(input_)

        self.assertTrue(torch.equal(first_output, second_output))

    def test_output_dropout_changes_training_outputs(self):
        encoder = encoders.LSTMEncoder(
            lstm_args(enc_output_dropout=0.5, enc_output_dropout_type="standard"))
        input_ = self.sample_input()

        encoder.train()
        first_output, _ = encoder(input_)
        second_output, _ = encoder(input_)

        self.assertFalse(torch.equal(first_output, second_output))
        self.assertEqual((5, 2, 6), tuple(first_output.shape))

    def test_output_dropout_is_inactive_in_eval_mode(self):
        encoder = encoders.LSTMEncoder(
            lstm_args(enc_output_dropout=0.5, enc_output_dropout_type="locked"))
        input_ = self.sample_input()

        encoder.eval()
        first_output, _ = encoder(input_)
        second_output, _ = encoder(input_)

        self.assertTrue(torch.equal(first_output, second_output))

    def test_locked_dropout_mask_is_constant_over_time(self):
        dropout = encoders.LockedDropout(0.5)
        input_ = torch.ones(6, 3, 8)

        torch.manual_seed(1)
        dropout.train()
        output = dropout(input_)

        mask = output != 0
        for t in range(1, output.size(0)):
            self.assertTrue(torch.equal(mask[0], mask[t]))
        self.assertTrue(torch.any(mask[0] != mask[0, 0].unsqueeze(0)))

    def test_single_layer_enc_dropout_warning(self):
        with self.assertWarnsRegex(UserWarning, "enc-dropout has no effect"):
            encoders.LSTMEncoder(lstm_args(enc_dropout=0.5))

    def test_warning_not_emitted_without_ineffective_dropout(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            encoders.LSTMEncoder(lstm_args(enc_dropout=0.))
        self.assertEqual([], caught)

    def test_warning_not_emitted_for_stacked_lstm_dropout(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            encoders.LSTMEncoder(lstm_args(enc_layers=2, enc_dropout=0.5))
        self.assertEqual([], caught)


if __name__ == "__main__":
    unittest.main()

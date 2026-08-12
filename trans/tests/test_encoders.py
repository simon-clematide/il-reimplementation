"""Unit tests for encoder components."""
import unittest

import torch

from trans.encoders import SinusoidalPositionalEmbedding


class TestSinusoidalPositionalEmbedding(unittest.TestCase):

    def test_positions_are_independent_across_batch_rows(self):
        padding_idx = 1
        embedding = SinusoidalPositionalEmbedding(
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
        embedding = SinusoidalPositionalEmbedding(
            embedding_dim=5,
            padding_idx=padding_idx,
            init_size=2,
        )
        tokens = torch.tensor([[2, 3, 4, 5]])

        positions = embedding(tokens)

        self.assertEqual((1, 4, 5), tuple(positions.shape))
        self.assertGreaterEqual(embedding.weights.size(0), padding_idx + 1 + tokens.size(1))


if __name__ == "__main__":
    unittest.main()

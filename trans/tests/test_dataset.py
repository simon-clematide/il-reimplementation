"""Unit tests for Dataset and Sample utilities."""
import unittest

import torch

from trans import utils


class TestSampleDeviceTransfer(unittest.TestCase):

    def test_sample_to_moves_every_tensor_field(self):
        sample = utils.Sample(
            input="ab",
            target="xy",
            encoded_input=torch.tensor([0, 1]),
            action_history=torch.tensor([0, 1]),
            alignment_history=torch.tensor([0, 1]),
            optimal_actions_mask=torch.tensor([[True, False]]),
            valid_actions_mask=torch.tensor([[True, True]]),
            encoded_features=torch.tensor([2, 3]),
        )

        sample.to("cpu")

        for attr in sample._tensor_attrs:
            with self.subTest(attr=attr):
                self.assertEqual(torch.device("cpu"), getattr(sample, attr).device)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_sample_to_moves_every_tensor_field_to_mps(self):
        sample = utils.Sample(
            input="ab",
            target="xy",
            encoded_input=torch.tensor([0, 1]),
            action_history=torch.tensor([0, 1]),
            alignment_history=torch.tensor([0, 1]),
            optimal_actions_mask=torch.tensor([[True, False]]),
            valid_actions_mask=torch.tensor([[True, True]]),
            encoded_features=torch.tensor([2, 3]),
        )

        sample.to("mps")

        for attr in sample._tensor_attrs:
            with self.subTest(attr=attr):
                self.assertEqual(torch.device("mps"), getattr(sample, attr).device)


if __name__ == "__main__":
    unittest.main()

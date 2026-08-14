"""Unit tests for Dataset and Sample utilities."""
import unittest

import torch

from trans import utils
from trans import vocabulary


class TestSampleDeviceTransfer(unittest.TestCase):

    def test_sample_to_moves_every_tensor_field(self):
        sample = utils.Sample(
            input="ab",
            target="xy",
            encoded_input=torch.tensor([0, 1]),
            action_history=torch.tensor([0, 1]),
            output_history=torch.tensor([0, 1]),
            alignment_history=torch.tensor([0, 1]),
            expert_action_costs=torch.tensor([[0., 1.]]),
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
            output_history=torch.tensor([0, 1]),
            alignment_history=torch.tensor([0, 1]),
            expert_action_costs=torch.tensor([[0., 1.]]),
            optimal_actions_mask=torch.tensor([[True, False]]),
            valid_actions_mask=torch.tensor([[True, True]]),
            encoded_features=torch.tensor([2, 3]),
        )

        sample.to("mps")

        for attr in sample._tensor_attrs:
            with self.subTest(attr=attr):
                self.assertEqual("mps", getattr(sample, attr).device.type)


class TestTrainingCollation(unittest.TestCase):

    def test_training_collate_uses_valid_padding_ids_for_histories(self):
        dataset = utils.Dataset([
            utils.Sample(
                input=["a"],
                target=["a"],
                encoded_input=torch.tensor([0, 4, 1]),
                action_history=torch.tensor([vocabulary.BEGIN_WORD, vocabulary.COPY]),
                output_history=torch.tensor([0, 4]),
                alignment_history=torch.tensor([0, 1]),
                expert_action_costs=torch.tensor([[0., 1.], [0., 1.]]),
                optimal_actions_mask=torch.tensor([[True, False], [True, False]]),
                valid_actions_mask=torch.tensor([[True, True], [True, True]]),
            ),
            utils.Sample(
                input=["b"],
                target=["b"],
                encoded_input=torch.tensor([0, 5, 6, 1]),
                action_history=torch.tensor([vocabulary.BEGIN_WORD]),
                output_history=torch.tensor([0]),
                alignment_history=torch.tensor([0]),
                expert_action_costs=torch.tensor([[0., 1.]]),
                optimal_actions_mask=torch.tensor([[True, False]]),
                valid_actions_mask=torch.tensor([[True, True]]),
            ),
        ])

        batch = next(iter(dataset.get_data_loader(
            is_training=True,
            batch_size=2,
            device="cpu",
            shuffle=False,
        )))

        self.assertGreaterEqual(batch.action_history.min().item(), 0)
        self.assertGreaterEqual(batch.output_history.min().item(), 0)
        self.assertGreaterEqual(batch.alignment_history.min().item(), 0)
        self.assertEqual(vocabulary.PAD, batch.action_history[1, 1].item())
        self.assertEqual(vocabulary.PAD, batch.output_history[1, 1].item())
        self.assertEqual(0, batch.alignment_history.view(2, 2)[1, 1].item())

    def test_training_collate_sanitizes_negative_legacy_sentinals(self):
        dataset = utils.Dataset([
            utils.Sample(
                input=["a"],
                target=["a"],
                encoded_input=torch.tensor([0, 4, 1]),
                action_history=torch.tensor([vocabulary.BEGIN_WORD, -1]),
                output_history=torch.tensor([0, -1]),
                alignment_history=torch.tensor([0, -1]),
                expert_action_costs=torch.tensor([[0., 1.], [0., 1.]]),
                optimal_actions_mask=torch.tensor([[True, False], [True, False]]),
                valid_actions_mask=torch.tensor([[True, True], [True, True]]),
            ),
        ])

        batch = next(iter(dataset.get_data_loader(
            is_training=True,
            batch_size=1,
            device="cpu",
            shuffle=False,
        )))

        self.assertEqual(vocabulary.PAD, batch.action_history[1, 0].item())
        self.assertEqual(vocabulary.PAD, batch.output_history[1, 0].item())
        self.assertEqual(0, batch.alignment_history[1].item())


if __name__ == "__main__":
    unittest.main()

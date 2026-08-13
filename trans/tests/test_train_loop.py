"""Unit tests for train-loop helpers."""
import argparse
import json
import os
import tempfile
import unittest

import torch

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
                dev_accuracy=0.0,
                train_accuracy=0.25,
            )

            with open(metadata_path) as f:
                metadata = json.load(f)

        self.assertEqual(0, metadata["epoch"])
        self.assertEqual(0.0, metadata["dev_accuracy"])
        self.assertEqual(0.25, metadata["train_accuracy"])
        self.assertEqual({"device": "cpu", "epochs": 1}, metadata["args"])
        self.assertIn("git_commit", metadata)

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

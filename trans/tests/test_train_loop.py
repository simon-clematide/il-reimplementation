"""Unit tests for train-loop helpers."""
import argparse
import json
import os
import tempfile
import unittest

from trans import train


class TestGradientAccumulation(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()

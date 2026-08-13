"""Unit tests for decoder diagnostics."""
import argparse
import csv
import json
import os
import tempfile
import unittest

import torch

from trans import diagnose
from trans import optimal_expert_substitutions
from trans import sed
from trans import transducer
from trans import vocabulary
from trans.actions import ConditionalCopy, ConditionalDel, ConditionalIns, ConditionalSub


class DiagnoseTests(unittest.TestCase):

    def test_logsumexp(self):
        self.assertAlmostEqual(0., diagnose.logsumexp([float("-inf"), 0.]))

    def test_format_tsv_value_rounds_floats(self):
        self.assertEqual("0.9286", diagnose.format_tsv_value(0.9285714285714286))
        self.assertEqual("copy(i)", diagnose.format_tsv_value("copy(i)"))

    def test_action_label(self):
        self.assertEqual("copy", diagnose.action_label(ConditionalCopy()))
        self.assertEqual("del", diagnose.action_label(ConditionalDel()))
        self.assertEqual("ins(a)", diagnose.action_label(ConditionalIns("a")))
        self.assertEqual("sub(a)", diagnose.action_label(ConditionalSub("a")))

    def test_action_label_at_state_includes_source_symbol(self):
        self.assertEqual(
            "copy(a)",
            diagnose.action_label_at_state("ab", 0, ConditionalCopy()),
        )
        self.assertEqual(
            "del(b)",
            diagnose.action_label_at_state("ab", 1, ConditionalDel()),
        )
        self.assertEqual(
            "sub(a->x)",
            diagnose.action_label_at_state("ab", 0, ConditionalSub("x")),
        )
        self.assertEqual(
            "ins(x)",
            diagnose.action_label_at_state("ab", 0, ConditionalIns("x")),
        )

    def test_load_model_args_adds_new_dropout_defaults_for_old_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "best.model.json")
            with open(metadata_path, "w") as f:
                json.dump({"args": {"device": "mps", "enc_type": "lstm"}}, f)

            args = diagnose.load_model_args(metadata_path, device="cpu")

        self.assertEqual("cpu", args.device)
        self.assertEqual(0., args.enc_output_dropout)
        self.assertEqual("locked", args.enc_output_dropout_type)

    def test_diagnose_writes_summary_and_step_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocabularies = vocabulary.Vocabularies()
            vocabularies.encode_input("a")
            vocabularies.encode_actions("b")
            vocabulary_path = os.path.join(tmpdir, "vocabulary.pkl")
            vocabularies.persist(vocabulary_path)

            sed_model = sed.StochasticEditDistance.build_sed("a", "b", copy_probability=None)
            sed_path = os.path.join(tmpdir, "sed.pkl")
            sed_model.to_pickle(sed_path)

            args = argparse.Namespace(
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
                dec_hidden_dim=4,
                dec_layers=1,
            )
            expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_model)
            model = transducer.Transducer(vocabularies, expert, args)
            model_path = os.path.join(tmpdir, "best.model")
            torch.save(model.state_dict(), model_path)

            metadata_path = os.path.join(tmpdir, "best.model.json")
            with open(metadata_path, "w") as f:
                json.dump({"args": vars(args)}, f)

            input_path = os.path.join(tmpdir, "test.tsv")
            with open(input_path, "w", encoding="utf8") as f:
                f.write("a\tb\n")

            output_dir = os.path.join(tmpdir, "diagnostics")
            diagnose.main(argparse.Namespace(
                model=model_path,
                metadata=metadata_path,
                vocabulary=vocabulary_path,
                sed_params=sed_path,
                input=input_path,
                output=output_dir,
                top_k_actions=3,
                errors_only=False,
                nfd=False,
                device="cpu",
            ))

            summary_path = os.path.join(output_dir, "diagnostics.tsv")
            steps_path = os.path.join(output_dir, "diagnostic_steps.tsv")
            self.assertTrue(os.path.exists(summary_path))
            self.assertTrue(os.path.exists(steps_path))

            with open(summary_path, encoding="utf8") as f:
                summary_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(steps_path, encoding="utf8") as f:
                step_rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(1, len(summary_rows))
            self.assertEqual("a", summary_rows[0]["source"])
            self.assertEqual("b", summary_rows[0]["gold"])
            self.assertIn("first_non_optimal_step", summary_rows[0])
            self.assertGreaterEqual(len(step_rows), 1)
            self.assertEqual("b", step_rows[0]["gold"])
            self.assertIn("oracle_actions", step_rows[0])
            self.assertIn("oracle_mass_prob", step_rows[0])
            self.assertIn("top_actions", step_rows[0])


if __name__ == "__main__":
    unittest.main()

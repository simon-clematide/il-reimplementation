"""Unit tests for decoder diagnostics."""
import argparse
import csv
import json
import math
import os
import tempfile
import unittest

import torch

from trans import diagnose
from trans import optimal_expert_substitutions
from trans import sed
from trans import transducer
from trans import utils
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

    def test_action_type(self):
        self.assertEqual("COPY", diagnose.action_type(ConditionalCopy()))
        self.assertEqual("DEL", diagnose.action_type(ConditionalDel()))
        self.assertEqual("INS", diagnose.action_type(ConditionalIns("a")))
        self.assertEqual("SUB", diagnose.action_type(ConditionalSub("a")))
        self.assertEqual("NONE", diagnose.action_type(None))

    def test_mapping_prediction_bucket(self):
        self.assertEqual(
            "predicted_correct_sub",
            diagnose.mapping_prediction_bucket(
                ConditionalSub("k"),
                ConditionalSub("k"),
            ),
        )
        self.assertEqual(
            "predicted_wrong_sub",
            diagnose.mapping_prediction_bucket(
                ConditionalSub("k"),
                ConditionalSub("t͡ʃ"),
            ),
        )
        self.assertEqual(
            "predicted_copy",
            diagnose.mapping_prediction_bucket(
                ConditionalSub("k"),
                ConditionalCopy(),
            ),
        )
        self.assertEqual(
            "predicted_sub",
            diagnose.mapping_prediction_bucket(
                ConditionalCopy(),
                ConditionalSub("k"),
            ),
        )

    def test_inference_replay_stats_tracks_expert_sub_to_model_copy(self):
        stats = diagnose.InferenceReplayStats()

        diagnose.update_inference_replay_stats(
            stats,
            ["c"],
            0,
            ConditionalCopy(),
            [ConditionalSub("k")],
            oracle_optimal=False,
            phase="after_first_deviation",
        )
        diagnose.update_inference_replay_stats(
            stats,
            ["a"],
            0,
            ConditionalCopy(),
            [ConditionalCopy()],
            oracle_optimal=True,
            phase="before_first_deviation",
        )

        self.assertEqual(2, stats.total_decisions)
        self.assertEqual(1, stats.optimal_decisions)
        self.assertEqual(2, stats.by_predicted_action["COPY"]["total"])
        self.assertEqual(1, stats.by_predicted_action["COPY"]["optimal"])
        self.assertEqual(1, stats.by_predicted_action["COPY"]["nonoptimal"])
        self.assertEqual(1, stats.expert_sub_model_action["COPY"])
        self.assertEqual(1, stats.expert_sub_model_copy_pairs["c->k"])
        self.assertEqual(
            1,
            stats.first_deviation["after_first_deviation"][
                "expert_sub_model_copy"],
        )

    def test_expert_mapping_distribution_rows(self):
        stats = diagnose.ExpertMappingStats()
        stats.counts[("o", "COPY", "o")] = 9
        stats.counts[("o", "SUB", "ɔ")] = 1
        stats.source_totals["o"] = 10

        rows = diagnose.expert_mapping_distribution_rows(stats)

        self.assertEqual(2, len(rows))
        copy_row = next(row for row in rows if row["expert_action"] == "COPY")
        self.assertEqual("o", copy_row["source"])
        self.assertEqual("o", copy_row["target"])
        self.assertEqual(9, copy_row["count"])
        self.assertAlmostEqual(0.9, copy_row["proportion_for_source"])

    def test_sub_copy_error_training_rows_join_training_and_sed_probs(self):
        replay_stats = diagnose.InferenceReplayStats()
        replay_stats.expert_sub_model_copy_pairs["o->ɔ"] = 5
        expert_stats = diagnose.ExpertMappingStats()
        expert_stats.counts[("o", "COPY", "o")] = 9
        expert_stats.counts[("o", "SUB", "ɔ")] = 1
        expert_stats.source_totals["o"] = 10
        sed_model = sed.StochasticEditDistance(sed.ParamDict(
            delta_sub={
                ("o", "o"): math.log(0.6),
                ("o", "ɔ"): math.log(0.3),
            },
            delta_del={"o": math.log(0.05)},
            delta_ins={"o": math.log(0.02), "ɔ": math.log(0.02)},
            delta_eos=math.log(0.01),
            source_alphabet=("o",),
            target_alphabet=("o", "ɔ"),
        ))

        rows = diagnose.sub_copy_error_training_rows(
            replay_stats,
            expert_stats,
            sed_model,
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("o->ɔ", rows[0]["test_error"])
        self.assertEqual(5, rows[0]["error_count"])
        self.assertEqual(9, rows[0]["train_copy"])
        self.assertEqual(1, rows[0]["train_same_sub"])
        self.assertAlmostEqual(0.1, rows[0]["train_same_sub_share"])
        self.assertAlmostEqual(
            0.6 / 0.9,
            rows[0]["sed_copy_probability_given_source"],
        )
        self.assertAlmostEqual(
            0.3 / 0.9,
            rows[0]["sed_sub_probability_given_source"],
        )

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

    def test_delete_regret_row_compares_delete_to_best_map_action(self):
        vocabularies = vocabulary.Vocabularies()
        vocabularies.encode_input("a")
        vocabularies.encode_actions(["a", "b"])
        params = sed.ParamDict(
            delta_sub={("a", "a"): torch.log(torch.tensor(0.10)).item(),
                       ("a", "b"): torch.log(torch.tensor(0.40)).item()},
            delta_del={"a": torch.log(torch.tensor(0.20)).item()},
            delta_ins={"a": torch.log(torch.tensor(0.10)).item(),
                       "b": torch.log(torch.tensor(0.10)).item()},
            delta_eos=torch.log(torch.tensor(0.10)).item(),
            source_alphabet=("a",),
            target_alphabet=("a", "b"),
        )
        sed_model = sed.StochasticEditDistance(params)
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
        sample = utils.Sample(["a"], ["b"])
        valid_actions_mask = torch.zeros(
            (1, 1, model.number_actions),
            dtype=torch.bool,
        )
        valid_actions_mask[0, 0, vocabulary.DELETE] = True
        valid_actions_mask[0, 0, vocabulary.COPY] = True
        sub_b_id = vocabularies.encode_unseen_action(ConditionalSub("b"))
        valid_actions_mask[0, 0, sub_b_id] = True
        logits = torch.zeros((1, 1, model.number_actions))
        log_probs = model.log_softmax(logits, valid_actions_mask)[0, 0]

        row = diagnose.delete_regret_row(
            model,
            sample,
            source_text="a",
            gold_text="b",
            prediction="b",
            correct=True,
            step=0,
            alignment=0,
            prediction_so_far=[],
            log_probs=log_probs,
            valid_actions_mask=valid_actions_mask,
        )

        self.assertEqual("a", row["source_symbol"])
        self.assertEqual("b", row["best_sed_map_target"])
        self.assertEqual("sub(a->b)", row["best_map_action"])
        self.assertGreater(row["delete_regret"], 0.)
        self.assertGreater(row["delete_damage"], 0.)

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
                training_input=input_path,
                output=output_dir,
                top_k_actions=3,
                errors_only=False,
                nfd=False,
                device="cpu",
            ))

            summary_path = os.path.join(output_dir, "diagnostics.tsv")
            steps_path = os.path.join(output_dir, "diagnostic_steps.tsv")
            action_type_confusion_path = os.path.join(
                output_dir, "action_type_confusion.tsv")
            emission_confusion_path = os.path.join(
                output_dir, "emission_confusion.tsv")
            mapping_bias_path = os.path.join(output_dir, "mapping_bias.tsv")
            inference_summary_path = os.path.join(
                output_dir, "inference_action_summary.tsv")
            inference_by_prediction_path = os.path.join(
                output_dir, "inference_action_by_prediction.tsv")
            expert_sub_model_action_path = os.path.join(
                output_dir, "expert_sub_model_action.tsv")
            expert_sub_model_copy_pairs_path = os.path.join(
                output_dir, "expert_sub_model_copy_pairs.tsv")
            first_deviation_path = os.path.join(
                output_dir, "first_deviation_diagnostics.tsv")
            delete_diagnostics_path = os.path.join(
                output_dir, "delete_diagnostics.tsv")
            expert_mapping_distribution_path = os.path.join(
                output_dir, "expert_mapping_distribution.tsv")
            sub_copy_error_training_path = os.path.join(
                output_dir, "sub_copy_error_training.tsv")
            self.assertTrue(os.path.exists(summary_path))
            self.assertTrue(os.path.exists(steps_path))
            self.assertTrue(os.path.exists(action_type_confusion_path))
            self.assertTrue(os.path.exists(emission_confusion_path))
            self.assertTrue(os.path.exists(mapping_bias_path))
            self.assertTrue(os.path.exists(inference_summary_path))
            self.assertTrue(os.path.exists(inference_by_prediction_path))
            self.assertTrue(os.path.exists(expert_sub_model_action_path))
            self.assertTrue(os.path.exists(expert_sub_model_copy_pairs_path))
            self.assertTrue(os.path.exists(first_deviation_path))
            self.assertTrue(os.path.exists(delete_diagnostics_path))
            self.assertTrue(os.path.exists(expert_mapping_distribution_path))
            self.assertTrue(os.path.exists(sub_copy_error_training_path))

            with open(summary_path, encoding="utf8") as f:
                summary_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(steps_path, encoding="utf8") as f:
                step_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(action_type_confusion_path, encoding="utf8") as f:
                action_type_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(emission_confusion_path, encoding="utf8") as f:
                emission_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(mapping_bias_path, encoding="utf8") as f:
                mapping_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(inference_summary_path, encoding="utf8") as f:
                inference_summary_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(inference_by_prediction_path, encoding="utf8") as f:
                inference_by_prediction_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(first_deviation_path, encoding="utf8") as f:
                first_deviation_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(delete_diagnostics_path, encoding="utf8") as f:
                delete_diagnostic_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(expert_mapping_distribution_path, encoding="utf8") as f:
                expert_mapping_rows = list(csv.DictReader(f, delimiter="\t"))
            with open(sub_copy_error_training_path, encoding="utf8") as f:
                sub_copy_training_rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(1, len(summary_rows))
            self.assertEqual("a", summary_rows[0]["source"])
            self.assertEqual("b", summary_rows[0]["gold"])
            self.assertIn("first_non_optimal_step", summary_rows[0])
            self.assertGreaterEqual(len(step_rows), 1)
            self.assertEqual("b", step_rows[0]["gold"])
            self.assertIn("previous_output", step_rows[0])
            self.assertIn("oracle_actions", step_rows[0])
            self.assertIn("oracle_mass_prob", step_rows[0])
            self.assertIn("top_actions", step_rows[0])
            self.assertGreaterEqual(len(action_type_rows), 1)
            self.assertIn("gold_action_type", action_type_rows[0])
            self.assertGreaterEqual(len(emission_rows), 1)
            self.assertIn("gold_action", emission_rows[0])
            self.assertEqual(
                {"reference_identity", "reference_nonidentity"},
                {row["reference_mapping"] for row in mapping_rows},
            )
            self.assertEqual(
                {"total_decisions", "optimal_decisions", "optimal_pct"},
                {row["metric"] for row in inference_summary_rows},
            )
            self.assertGreaterEqual(len(inference_by_prediction_rows), 1)
            self.assertIn(
                "predicted_action_type",
                inference_by_prediction_rows[0],
            )
            self.assertEqual(
                {"before_first_deviation", "after_first_deviation"},
                {row["phase"] for row in first_deviation_rows},
            )
            if delete_diagnostic_rows:
                self.assertIn("delete_regret", delete_diagnostic_rows[0])
                self.assertIn("best_map_action", delete_diagnostic_rows[0])
            self.assertGreaterEqual(len(expert_mapping_rows), 1)
            self.assertIn("proportion_for_source", expert_mapping_rows[0])
            self.assertIn(
                "train_same_sub_share",
                sub_copy_training_rows[0]
                if sub_copy_training_rows else {
                    "train_same_sub_share": "",
                },
            )


if __name__ == "__main__":
    unittest.main()

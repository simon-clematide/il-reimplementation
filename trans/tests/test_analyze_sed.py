"""Unit tests for SED analysis CLI helpers."""
import argparse
import os
import tempfile
import unittest

import numpy as np

from trans import analyze_sed
from trans import sed
from trans import utils


class TestAnalyzeSed(unittest.TestCase):

    def setUp(self):
        params = sed.ParamDict(
            delta_sub={("a", "a"): np.log(0.10), ("a", "b"): np.log(0.01)},
            delta_del={"a": np.log(0.40)},
            delta_ins={"b": np.log(0.40)},
            delta_eos=np.log(0.09),
            source_alphabet=("a",),
            target_alphabet=("a", "b"),
        )
        self.sed = sed.StochasticEditDistance(params)

    def test_analyze_sample_scores(self):
        sample = utils.Sample(input="a", target="b")

        analysis = analyze_sed.analyze_sample(self.sed, sample, line_number=7)

        self.assertEqual(7, analysis.line_number)
        self.assertEqual("a", analysis.source)
        self.assertEqual("b", analysis.target)
        self.assertTrue(np.isclose(
            -self.sed.stochastic_distance("a", "b"),
            analysis.stochastic_surprisal,
        ))
        self.assertTrue(np.isclose(
            -self.sed.viterbi_distance("a", "b"),
            analysis.viterbi_surprisal,
        ))
        self.assertTrue(np.isclose(
            analysis.stochastic_logp - analysis.viterbi_logp,
            analysis.alignment_ambiguity,
        ))
        self.assertIn("del(a)", analysis.alignment)
        self.assertIn("ins(b)", analysis.alignment)

    def test_write_analyses_outputs_header_and_rows(self):
        analysis = analyze_sed.analyze_sample(
            self.sed,
            utils.Sample(input="a", target="b"),
            line_number=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "analysis.tsv")
            analyze_sed.write_analyses([analysis], output)
            with open(output, encoding="utf8") as f:
                lines = f.readlines()

        self.assertEqual(2, len(lines))
        self.assertTrue(lines[0].startswith("line_number\tsource\ttarget"))
        self.assertIn("\ta\tb\t", lines[1])

    def test_read_tsv_uses_first_two_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.tsv")
            with open(path, "w", encoding="utf8") as w:
                w.write("a\tb\tignored\n")

            samples = analyze_sed.read_tsv(path)

        self.assertEqual([utils.Sample(input="a", target="b")], samples)

    def test_mapping_summary_rows_reports_top_source_mappings(self):
        rows = analyze_sed.mapping_summary_rows(self.sed, limit=2)

        self.assertEqual(["a", "a"], [row["source"] for row in rows])
        self.assertEqual(["a", "b"], [row["target"] for row in rows])
        self.assertTrue(rows[0]["identity"])
        self.assertFalse(rows[1]["identity"])
        self.assertTrue(np.isclose(
            0.10 / 0.11,
            rows[0]["map_probability_given_source"],
        ))
        self.assertTrue(np.isclose(
            0.10 / 0.11,
            rows[0]["identity_map_mass_given_source"],
        ))
        self.assertTrue(np.isclose(
            0.01 / 0.11,
            rows[0]["nonidentity_map_mass_given_source"],
        ))

    def test_map_mass_summary_rows_reports_diagonal_and_nondiagonal_mass(self):
        rows = analyze_sed.map_mass_summary_rows(self.sed)
        values = {row["metric"]: row["value"] for row in rows}

        self.assertTrue(np.isclose(0.11, values["map_event_mass"]))
        self.assertTrue(np.isclose(0.10, values["diagonal_map_mass"]))
        self.assertTrue(np.isclose(0.01, values["nondiagonal_map_mass"]))
        self.assertTrue(np.isclose(
            0.10 / 0.11,
            values["diagonal_map_mass_given_map"],
        ))

    def test_copy_preference_rows_reports_overlap_sorted_by_identity_rate(self):
        params = sed.ParamDict(
            delta_sub={
                ("a", "a"): np.log(0.30),
                ("a", "b"): np.log(0.10),
                ("e", "e"): np.log(0.05),
                ("e", "b"): np.log(0.20),
            },
            delta_del={"a": np.log(0.05), "e": np.log(0.05)},
            delta_ins={"a": np.log(0.05), "b": np.log(0.05), "e": np.log(0.05)},
            delta_eos=np.log(0.10),
            source_alphabet=("a", "e"),
            target_alphabet=("a", "b", "e"),
        )
        sed_ = sed.StochasticEditDistance(params)

        rows = analyze_sed.copy_preference_rows(sed_)

        self.assertEqual(["e", "a"], [row["source"] for row in rows])
        self.assertEqual("b", rows[0]["best_nonidentity_target"])
        self.assertFalse(rows[0]["best_target_is_identity"])
        self.assertTrue(np.isclose(
            0.05 / 0.25,
            rows[0]["identity_probability_given_source"],
        ))
        self.assertTrue(rows[1]["best_target_is_identity"])

    def test_main_writes_event_and_mapping_tables_without_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sed_path = os.path.join(tmpdir, "sed.pkl")
            events_path = os.path.join(tmpdir, "events.tsv")
            mappings_path = os.path.join(tmpdir, "mappings.tsv")
            map_mass_path = os.path.join(tmpdir, "map-mass.tsv")
            copy_preference_path = os.path.join(tmpdir, "copy-preference.tsv")
            self.sed.to_pickle(sed_path)

            analyze_sed.main(argparse.Namespace(
                sed_params=sed_path,
                input=None,
                output=None,
                sort_by="stochastic_surprisal",
                descending=True,
                limit=None,
                nfd=False,
                events_output=events_path,
                mappings_output=mappings_path,
                map_mass_output=map_mass_path,
                copy_preference_output=copy_preference_path,
                top_k_mappings=2,
            ))

            with open(events_path, encoding="utf8") as f:
                events_text = f.read()
            with open(mappings_path, encoding="utf8") as f:
                mappings_text = f.read()
            with open(map_mass_path, encoding="utf8") as f:
                map_mass_text = f.read()
            with open(copy_preference_path, encoding="utf8") as f:
                copy_preference_text = f.read()

        self.assertIn("event\tsource\ttarget", events_text)
        self.assertIn("map\ta\tb", events_text)
        self.assertIn("source\trank\ttarget", mappings_text)
        self.assertIn("map_probability_given_source", mappings_text)
        self.assertIn("a\t1\ta", mappings_text)
        self.assertIn("diagonal_map_mass", map_mass_text)
        self.assertIn("identity_probability_given_source", copy_preference_text)


if __name__ == "__main__":
    unittest.main()

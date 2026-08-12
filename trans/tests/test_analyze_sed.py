"""Unit tests for SED analysis CLI helpers."""
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
            delta_sub={("a", "b"): np.log(0.01)},
            delta_del={"a": np.log(0.40)},
            delta_ins={"b": np.log(0.40)},
            delta_eos=np.log(0.19),
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


if __name__ == "__main__":
    unittest.main()

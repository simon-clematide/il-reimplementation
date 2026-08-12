"""Unit tests for sed.py."""
import logging
import unittest

import numpy as np

from trans.actions import Del, Sub, Ins
from trans import sed
from trans.tests import test_optimal_expert_substitutions


class TestTransducer(unittest.TestCase):

    def setUp(self) -> None:

        self.source_alphabet1 = list("abcdefg")
        self.target_alphabet1 = list("fghijk")

        self.smart_sed = sed.StochasticEditDistance.build_sed(
            self.source_alphabet1, self.target_alphabet1)

    def test_sed_random_initialization(self):

        sed_ = sed.StochasticEditDistance.build_sed(
            self.source_alphabet1, self.target_alphabet1, copy_probability=None)
        eos_weight = sed_.delta_eos

        for weight_dict in ("delta_del", "delta_ins", "delta_sub"):
            for weight in getattr(sed_, weight_dict).values():
                self.assertTrue(np.isclose(eos_weight, weight))

    def test_sed_copy_biased_initialization(self):

        sed_ = sed.StochasticEditDistance.build_sed(
            self.source_alphabet1, self.target_alphabet1)
        eos_weight = sed_.delta_eos

        for weight_dict in ("delta_del", "delta_ins"):
            for weight in getattr(sed_, weight_dict).values():
                self.assertTrue(np.isclose(eos_weight, weight))

        for (x, y), weight in sed_.delta_sub.items():
            if x == y:
                self.assertFalse(np.isclose(eos_weight, weight))
            else:
                self.assertTrue(np.isclose(eos_weight, weight))

    def test_viterbi_decoding(self):

        best_edits, distance = self.smart_sed.viterbi_distance(
            source="affa", target="iffig", with_alignment=True)

        self.assertTrue(np.isclose(-26.7633, distance))
        self.assertEqual("iffig", self.replay("affa", best_edits))
        self.assertTrue(np.isclose(
            self.smart_sed.alignment_log_probability(best_edits),
            distance,
        ))

    def test_viterbi_traceback_uses_transition_weights(self):
        params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.01)},
            delta_del={"a": np.log(0.40)},
            delta_ins={"b": np.log(0.40)},
            delta_eos=np.log(0.19),
        )
        sed_ = sed.StochasticEditDistance(params)

        alignment, score = sed_.viterbi_distance(
            "a", "b", with_alignment=True)

        self.assertEqual(2, len(alignment))
        self.assertTrue(any(isinstance(action, Del) for action in alignment))
        self.assertTrue(any(isinstance(action, Ins) for action in alignment))
        self.assertTrue(np.isclose(
            np.log(0.40) + np.log(0.40) + np.log(0.19),
            score,
        ))
        self.assertTrue(np.isclose(
            sed_.alignment_log_probability(alignment),
            score,
        ))

    def test_viterbi_alignment_score_consistency(self):
        pairs = [
            ("", ""),
            ("a", ""),
            ("", "f"),
            ("a", "f"),
            ("affa", "iffig"),
            ("abc", "fgh"),
        ]
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                alignment, score = self.smart_sed.viterbi_distance(
                    source, target, with_alignment=True)
                self.assertTrue(np.isclose(
                    self.smart_sed.alignment_log_probability(alignment),
                    score,
                ))

    def test_forward_backward_eos_invariant(self):
        pairs = [
            ("", ""),
            ("a", ""),
            ("", "f"),
            ("a", "f"),
            ("abc", "fgh"),
        ]
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                alpha = self.smart_sed.forward_evaluate(source, target)
                beta = self.smart_sed.backward_evaluate(source, target)
                self.assertTrue(np.isclose(alpha[-1, -1], beta[0, 0]))

    @staticmethod
    def replay(source, alignment):
        output = []
        source_index = 0
        for action in alignment:
            if isinstance(action, Sub):
                assert action.old == source[source_index]
                source_index += 1
                output.append(action.new)
            elif isinstance(action, Del):
                assert action.old == source[source_index]
                source_index += 1
            elif isinstance(action, Ins):
                output.append(action.new)
            else:
                raise AssertionError(f"Unexpected action: {action}")
        assert source_index == len(source)
        return "".join(output)

    def test_viterbi_alignment_leading_insertions(self):
        edits, _ = self.smart_sed.viterbi_distance(
            source="", target="abc", with_alignment=True)

        self.assertListEqual([Ins("a"), Ins("b"), Ins("c")], edits)
        self.assertEqual("abc", self.replay("", edits))

    def test_viterbi_alignment_leading_deletions(self):
        edits, _ = self.smart_sed.viterbi_distance(
            source="abc", target="", with_alignment=True)

        self.assertListEqual([Del("a"), Del("b"), Del("c")], edits)
        self.assertEqual("", self.replay("abc", edits))

    def test_em_accepts_empty_source_and_target(self):
        sed_ = sed.StochasticEditDistance.build_sed("abc", "xyz")

        sed_.update_model(["", "abc"], ["xyz", ""], iterations=1)

        self.assertTrue(np.isfinite(sed_.log_likelihood(["", "abc"], ["xyz", ""])))

    def test_stochastic_decoding(self):

        distance = self.smart_sed.stochastic_distance(
            source="affa", target="iffig")

        self.assertTrue(np.isclose(-26.05741, distance))

    def test_em(self):

        input_pairs = [
            ("abby", "a b i"), ("abidjan", "a b i d ʒ ɑ"),
            ("abject", "a b ʒ ɛ k t"), ("abolir", "a b ɔ l i ʁ"),
            ("abonnement", "a b ɔ n m ɑ")
        ]

        sources, targets = zip(*input_pairs)

        source_alphabet = {c for source in sources for c in source}
        target_alphabet = {c for target in targets for c in target}

        sed_ = sed.StochasticEditDistance.build_sed(
            source_alphabet, target_alphabet)

        o = sed_.stochastic_distance(sources[1], targets[1])
        logging.info(o)

        before_ll = sed_.log_likelihood(sources, targets)
        sed_.update_model(sources, targets, iterations=1)
        after_ll = sed_.log_likelihood(sources, targets)
        self.assertTrue(before_ll <= after_ll)

    def test_fit_from_data(self):

        input_lines = [
            "abby\ta b i", "abidjan\ta b i d ʒ ɑ", "abject\ta b ʒ ɛ k t",
            "abolir\ta b ɔ l i ʁ", "abonnement\ta b ɔ n m ɑ"
        ]

        data = map(test_optimal_expert_substitutions.to_sample, input_lines)
        sed_ = sed.StochasticEditDistance.fit_from_data(data, em_iterations=1)
        logging.info(sed_.params)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format="%(levelname)s: %(message)s")
    TestTransducer().run()

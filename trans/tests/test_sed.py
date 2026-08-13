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

    def test_strict_em_does_not_decrease_likelihood(self):
        input_pairs = [("ab", "ac"), ("ba", "ca"), ("a", "a")]
        sources, targets = zip(*input_pairs)
        sed_ = sed.StochasticEditDistance.build_sed("ab", "ac", copy_probability=None)

        before_ll = sed_.log_likelihood(sources, targets)
        sed_.em(sources, targets, iterations=1, mode="strict")
        after_ll = sed_.log_likelihood(sources, targets)

        self.assertGreaterEqual(after_ll + 1e-10, before_ll)

    def test_strict_m_step_uses_normalized_expected_counts(self):
        params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.25)},
            delta_del={"a": np.log(0.25)},
            delta_ins={"b": np.log(0.25)},
            delta_eos=np.log(0.25),
        )
        sed_ = sed.StochasticEditDistance(params)
        gammas = sed.ParamDict(
            delta_sub={("a", "b"): np.log(2.)},
            delta_del={"a": np.log(1.)},
            delta_ins={"b": np.log(1.)},
            delta_eos=np.log(2.),
        )

        sed_.m_step(gammas, damping=1.)

        self.assertTrue(np.isclose(np.exp(sed_.delta_sub[("a", "b")]), 2 / 6))
        self.assertTrue(np.isclose(np.exp(sed_.delta_del["a"]), 1 / 6))
        self.assertTrue(np.isclose(np.exp(sed_.delta_ins["b"]), 1 / 6))
        self.assertTrue(np.isclose(np.exp(sed_.delta_eos), 2 / 6))

    def test_damped_m_step_interpolates_in_probability_space(self):
        params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.25)},
            delta_del={"a": np.log(0.25)},
            delta_ins={"b": np.log(0.25)},
            delta_eos=np.log(0.25),
        )
        sed_ = sed.StochasticEditDistance(params)
        gammas = sed.ParamDict(
            delta_sub={("a", "b"): np.log(2.)},
            delta_del={"a": np.log(1.)},
            delta_ins={"b": np.log(1.)},
            delta_eos=np.log(2.),
        )

        sed_.m_step(gammas, damping=0.5)

        self.assertTrue(np.isclose(
            np.exp(sed_.delta_sub[("a", "b")]),
            0.5 * (2 / 6) + 0.5 * 0.25,
        ))
        self.assertTrue(np.isclose(
            np.exp(sed_.delta_del["a"]),
            0.5 * (1 / 6) + 0.5 * 0.25,
        ))
        self.assertTrue(np.isclose(0., sed_.params.sum()))

    def test_damping_one_matches_strict_em(self):
        input_pairs = [("a", "b"), ("aa", "bb")]
        sources, targets = zip(*input_pairs)
        params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.25)},
            delta_del={"a": np.log(0.25)},
            delta_ins={"b": np.log(0.25)},
            delta_eos=np.log(0.25),
        )
        strict_sed = sed.StochasticEditDistance(sed.ParamDict.from_params(params))
        damped_sed = sed.StochasticEditDistance(sed.ParamDict.from_params(params))

        strict_sed.em(sources, targets, iterations=1, mode="strict")
        damped_sed.em(sources, targets, iterations=1,
                      mode="damped", damping=1.)

        self.assert_params_close(strict_sed.params, damped_sed.params)

    def test_interpolate_log_probabilities_matches_analytical_result(self):
        interpolated = sed.StochasticEditDistance.interpolate_log_probabilities(
            np.log(0.2), np.log(0.6), damping=0.75)

        self.assertTrue(np.isclose(np.log(0.5), interpolated))

    def test_zero_em_probability_survives_damping(self):
        interpolated = sed.StochasticEditDistance.interpolate_log_probabilities(
            old_value=np.log(0.4), em_value=sed.LOG_ZERO, damping=0.9)

        self.assertTrue(np.isclose(np.log(0.04), interpolated))

    def test_damped_parameters_are_normalized(self):
        old_params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.20)},
            delta_del={"a": np.log(0.30)},
            delta_ins={"b": np.log(0.10)},
            delta_eos=np.log(0.40),
        )
        em_params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.50)},
            delta_del={"a": np.log(0.20)},
            delta_ins={"b": np.log(0.20)},
            delta_eos=np.log(0.10),
        )

        for damping in (0.1, 0.5, 0.9):
            with self.subTest(damping=damping):
                damped = sed.StochasticEditDistance.damp_parameters(
                    old_params, em_params, damping)
                self.assertTrue(np.isclose(0., damped.sum()))

    def test_m_step_rejects_zero_expected_count(self):
        params = sed.ParamDict(
            delta_sub={("a", "b"): np.log(0.25)},
            delta_del={"a": np.log(0.25)},
            delta_ins={"b": np.log(0.25)},
            delta_eos=np.log(0.25),
        )
        sed_ = sed.StochasticEditDistance(params)
        empty_gammas = sed.ParamDict.zeros_like(params)

        with self.assertRaisesRegex(ValueError, "zero expected count"):
            sed_.m_step(empty_gammas)

        self.assertTrue(np.isclose(0., sed_.params.sum()))
        self.assertTrue(np.isfinite(sed_.delta_eos))

    def test_em_mode_and_damping_are_validated(self):
        sed_ = sed.StochasticEditDistance.build_sed("a", "b", copy_probability=None)

        with self.assertRaises(ValueError):
            sed_.em(["a"], ["b"], iterations=1, mode="unknown")
        with self.assertRaises(ValueError):
            sed_.em(["a"], ["b"], iterations=1, damping=0.)

    def test_em_rejects_empty_or_mismatched_corpus(self):
        sed_ = sed.StochasticEditDistance.build_sed("a", "b", copy_probability=None)

        with self.assertRaisesRegex(ValueError, "empty corpus"):
            sed_.em([], [], iterations=1)
        with self.assertRaisesRegex(ValueError, "same number"):
            sed_.em(["a"], [], iterations=1)

    def test_fit_from_data(self):

        input_lines = [
            "abby\ta b i", "abidjan\ta b i d ʒ ɑ", "abject\ta b ʒ ɛ k t",
            "abolir\ta b ɔ l i ʁ", "abonnement\ta b ɔ n m ɑ"
        ]

        data = map(test_optimal_expert_substitutions.to_sample, input_lines)
        sed_ = sed.StochasticEditDistance.fit_from_data(data, em_iterations=1)
        logging.info(sed_.params)

    def assert_params_close(self, left: sed.ParamDict, right: sed.ParamDict):
        self.assertEqual(left.delta_sub.keys(), right.delta_sub.keys())
        self.assertEqual(left.delta_del.keys(), right.delta_del.keys())
        self.assertEqual(left.delta_ins.keys(), right.delta_ins.keys())
        for key in left.delta_sub:
            self.assertTrue(np.isclose(left.delta_sub[key], right.delta_sub[key]))
        for key in left.delta_del:
            self.assertTrue(np.isclose(left.delta_del[key], right.delta_del[key]))
        for key in left.delta_ins:
            self.assertTrue(np.isclose(left.delta_ins[key], right.delta_ins[key]))
        self.assertTrue(np.isclose(left.delta_eos, right.delta_eos))


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format="%(levelname)s: %(message)s")
    TestTransducer().run()

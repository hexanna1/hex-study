import math
import unittest

import pattern_scoring


def _rows(samples):
    return [
        {"experiment": "e1", "move": move, "candidate_key_local": key, "mean_winrate": winrate, "n": 1}
        for move, key, winrate in samples
    ]


def _calibrate(with_rows, without_rows, roots=(0.52, 0.5)):
    rows, _ = pattern_scoring.apply_ablation_calibration(
        summary_rows_with=with_rows,
        summary_rows_without=without_rows,
        root_ablation_by_experiment={"e1": {
            "with_root_winrate": roots[0], "without_root_winrate": roots[1],
        }},
    )
    return {row["candidate_key_local"]: row for row in rows}


class PatternScoringTests(unittest.TestCase):
    def test_ablation_groups_orbits_and_cancels_independent_root_logit_offsets(self):
        with_rows = _rows([("m1", "pass_proxy", 0.4), ("m13", "a", 0.62), ("n13", "a", 0.58), ("l13", "b", 0.55)])
        without_rows = _rows([("m1", "pass_proxy", 0.38), ("m13", "a", 0.56), ("n13", "a", 0.54), ("l13", "b", 0.49)])
        baseline = _calibrate(with_rows, without_rows)

        def shifted(winrate, offset):
            return 1 / (1 + math.exp(-(math.log(winrate / (1 - winrate)) + offset)))

        shifted_rows = _calibrate(
            [{**row, "mean_winrate": shifted(row["mean_winrate"], 0.7)} for row in reversed(with_rows)],
            [{**row, "mean_winrate": shifted(row["mean_winrate"], -0.4)} for row in reversed(without_rows)],
            roots=(shifted(0.52, 0.7), shifted(0.5, -0.4)),
        )
        self.assertEqual(set(baseline), {"pass_proxy", "a", "b"})
        self.assertEqual(baseline["a"]["n"], 2)
        self.assertEqual(baseline["a"]["move"], "m13")
        self.assertAlmostEqual(baseline["a"]["mean_winrate"], 0.6)
        for key in baseline:
            self.assertAlmostEqual(baseline[key]["corrected_value"], shifted_rows[key]["corrected_value"])
        self.assertAlmostEqual(baseline["pass_proxy"]["corrected_value"], 0.0)

    def test_generic_move_value_keeps_real_move_above_pass_despite_weaker_interaction(self):
        rows = _calibrate(
            _rows([("m1", "pass_proxy", 0.25), ("a1", "a", 0.6), ("b1", "b", 0.7)]),
            _rows([("m1", "pass_proxy", 0.2), ("a1", "a", 0.55), ("b1", "b", 0.55)]),
            roots=(0.5, 0.5),
        )
        self.assertLess(rows["a"]["ablation_interaction_lift_logit"], rows["pass_proxy"]["ablation_interaction_lift_logit"])
        self.assertGreater(rows["a"]["corrected_value"], 0.0)
        self.assertAlmostEqual(rows["pass_proxy"]["corrected_value"], 0.0)
        self.assertGreater(rows["b"]["corrected_value"], rows["a"]["corrected_value"])

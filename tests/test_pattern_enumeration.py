import unittest

import pattern_enumeration as pattern_enumerate


class PatternEnumerationTests(unittest.TestCase):
    def test_enumerate_patterns_small_exact_catalog(self):
        catalog = pattern_enumerate.enumerate_patterns(max_moves=2, max_pair_delta=4)
        self.assertEqual(catalog["max_moves"], 2)
        self.assertEqual(catalog["total_patterns"], 5)
        self.assertEqual(
            catalog["counts_by_family"],
            [
                {"plus": 0, "minus": 1, "to_play": "blue", "count": 1},
                {"plus": 1, "minus": 0, "to_play": "red", "count": 1},
                {"plus": 1, "minus": 1, "to_play": "red", "count": 3},
            ],
        )
        patterns = [row["pattern"] for row in catalog["patterns"]]
        self.assertEqual(
            patterns,
            [
                "+[]-[0,0]",
                "+[0,0]-[]",
                "+[0,0]-[0,1]",
                "+[0,0]-[0,2]",
                "+[0,0]-[1,-2]",
            ],
        )
        by_pattern = {row["pattern"]: row for row in catalog["patterns"]}
        self.assertEqual(
            by_pattern["+[]-[0,0]"]["hexworld_21"],
            "https://hexworld.org/board/#21c1,k11",
        )
        self.assertEqual(by_pattern["+[]-[0,0]"]["to_play"], "blue")
        self.assertEqual(by_pattern["+[]-[0,0]"]["candidate_Δ_max"], 4)
        self.assertEqual(
            by_pattern["+[0,0]-[]"]["hexworld_21"],
            "https://hexworld.org/board/#21c1,k11:p",
        )
        self.assertEqual(by_pattern["+[0,0]-[]"]["to_play"], "red")
        self.assertEqual(by_pattern["+[0,0]-[]"]["candidate_Δ_max"], 4)
        self.assertEqual(
            by_pattern["+[0,0]-[0,1]"]["hexworld_21"],
            "https://hexworld.org/board/#21c1,k10k11",
        )
        self.assertEqual(by_pattern["+[0,0]-[0,1]"]["to_play"], "red")
        self.assertEqual(by_pattern["+[0,0]-[0,1]"]["candidate_Δ_max"], 4)

    def test_max_pair_delta_filters_longer_two_stone_patterns(self):
        catalog = pattern_enumerate.enumerate_patterns(
            max_moves=2,
            max_pair_delta=1,
            candidate_Δ_floor=1,
        )
        self.assertEqual(catalog["total_patterns"], 3)
        patterns = [row["pattern"] for row in catalog["patterns"]]
        self.assertEqual(
            patterns,
            [
                "+[]-[0,0]",
                "+[0,0]-[]",
                "+[0,0]-[0,1]",
            ],
        )

    def test_candidate_delta_uses_pattern_span_and_clamps_to_requested_interval(self):
        self.assertEqual(
            pattern_enumerate._candidate_Δ_max_for_pattern(
                plus=((0, 0),),
                minus=(),
                candidate_Δ_floor=4,
                multi_stone_candidate_Δ_cap=7,
                max_pair_delta=16,
                local_hexhex_side=7,
            ),
            16,
        )
        self.assertEqual(
            pattern_enumerate._candidate_Δ_max_for_pattern(
                plus=((0, 0),),
                minus=((0, 3),),
                candidate_Δ_floor=4,
                multi_stone_candidate_Δ_cap=7,
                max_pair_delta=16,
                local_hexhex_side=7,
            ),
            7,
        )
        self.assertEqual(
            pattern_enumerate._candidate_Δ_max_for_pattern(
                plus=((0, 0),),
                minus=((0, 4),),
                candidate_Δ_floor=4,
                multi_stone_candidate_Δ_cap=7,
                max_pair_delta=16,
                local_hexhex_side=7,
            ),
            4,
        )
        self.assertEqual(
            pattern_enumerate._candidate_Δ_max_for_pattern(
                plus=((0, 0),),
                minus=(),
                candidate_Δ_floor=4,
                multi_stone_candidate_Δ_cap=7,
                max_pair_delta=64,
                local_hexhex_side=9,
            ),
            57,
        )



if __name__ == "__main__":
    unittest.main()

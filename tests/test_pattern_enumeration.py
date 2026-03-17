import unittest

import pattern_enumeration


class PatternEnumerationTests(unittest.TestCase):
    def test_two_stone_catalog_is_complete_modulo_symmetry_and_preserves_side_to_play(self):
        # The three inequivalent two-stone distances <= 4 are 1, 3, and 4.
        catalog = pattern_enumeration.enumerate_patterns(max_moves=2, max_pair_delta=4)
        self.assertEqual([row["pattern"] for row in catalog["patterns"]], [
            "+[]-[0,0]", "+[0,0]-[]", "+[0,0]-[0,1]", "+[0,0]-[0,2]", "+[0,0]-[1,-2]",
        ])
        by_pattern = {row["pattern"]: row for row in catalog["patterns"]}
        self.assertEqual(by_pattern["+[]-[0,0]"]["hexworld_21"], "https://hexworld.org/board/#21c1,k11")
        self.assertEqual(by_pattern["+[0,0]-[]"]["hexworld_21"], "https://hexworld.org/board/#21c1,k11:p")
        self.assertEqual(by_pattern["+[0,0]-[0,1]"]["hexworld_21"], "https://hexworld.org/board/#21c1,k10k11")

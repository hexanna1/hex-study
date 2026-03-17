import unittest

from pattern_notation import (
    LabeledPattern,
    UnlabeledPattern,
    canonicalize,
    format_pattern,
    parse_pattern,
)


class PatternNotationTests(unittest.TestCase):
    def assert_parse_fails(self, s: str) -> None:
        with self.assertRaises(ValueError):
            parse_pattern(s)

    def test_parse_labeled_accepts_spaces_and_noncanonical_order(self):
        p = parse_pattern(" + [ 1 , 0 : 0 , 0 ] - [ 2 , -1 ] ")
        self.assertIsInstance(p, LabeledPattern)
        self.assertEqual(p.plus, ((1, 0), (0, 0)))
        self.assertEqual(p.minus, ((2, -1),))

    def test_parse_unlabeled_accepts_empty_block(self):
        p = parse_pattern("[0,0][]")
        self.assertIsInstance(p, UnlabeledPattern)
        self.assertEqual(p.a, ((0, 0),))
        self.assertEqual(p.b, ())

    def test_parse_rejects_duplicate_within_block(self):
        self.assert_parse_fails("+[0,0:0,0]-[]")
        self.assert_parse_fails("[1,2:1,2][]")

    def test_parse_rejects_cross_block_overlap(self):
        self.assert_parse_fails("+[0,0]-[0,0]")
        self.assert_parse_fails("[0,0][0,0]")

    def test_parse_rejects_shape_errors(self):
        self.assert_parse_fails("")
        self.assert_parse_fails("[]")
        self.assert_parse_fails("+[]")
        self.assert_parse_fails("+[]-")
        self.assert_parse_fails("+[0,0]-[1,1]x")
        self.assert_parse_fails("[0,0][1,1][2,2]")

    def test_canonicalize_translation_invariance_labeled(self):
        p1 = parse_pattern("+[0,0:1,0]-[0,1]")
        p2 = parse_pattern("+[10,-2:11,-2]-[10,-1]")
        c1 = format_pattern(canonicalize(p1))
        c2 = format_pattern(canonicalize(p2))
        self.assertEqual(c1, c2)

    def test_canonicalize_swap_invariance_unlabeled(self):
        p1 = parse_pattern("[0,0:1,0][0,1]")
        p2 = parse_pattern("[0,1][0,0:1,0]")
        c1 = format_pattern(canonicalize(p1))
        c2 = format_pattern(canonicalize(p2))
        self.assertEqual(c1, c2)

    def test_canonicalize_d6_rotation_reflection_invariance(self):
        p1 = parse_pattern("+[0,0:1,0]-[0,1]")
        p2 = parse_pattern("+[0,0:0,1]-[-1,1]")
        p3 = parse_pattern("+[0,0:1,-1]-[1,0]")
        self.assertEqual(format_pattern(canonicalize(p1)), format_pattern(canonicalize(p2)))
        self.assertEqual(format_pattern(canonicalize(p1)), format_pattern(canonicalize(p3)))

    def test_prompt_single_and_pair_patterns(self):
        # Single stone
        self.assertEqual(
            format_pattern(canonicalize(parse_pattern("+[0,0]-[]"))),
            "+[0,0]-[]",
        )
        # Adjacent pair (same color) and opposite color
        self.assertEqual(
            format_pattern(canonicalize(parse_pattern("+[0,0:1,0]-[]"))),
            "+[0,0:0,1]-[]",
        )
        self.assertEqual(
            format_pattern(canonicalize(parse_pattern("+[0,0]-[1,0]"))),
            "+[0,0]-[0,1]",
        )
        # Bridge pair (Delta3) same color and opposite color
        self.assertEqual(
            format_pattern(canonicalize(parse_pattern("+[0,0:1,1]-[]"))),
            "+[0,0:1,-2]-[]",
        )
        self.assertEqual(
            format_pattern(canonicalize(parse_pattern("+[0,0]-[1,1]"))),
            "+[0,0]-[1,-2]",
        )

    def test_prompt_two_red_adjacent_to_one_blue(self):
        # two plus stones both adjacent to one minus stone at origin
        p = parse_pattern("+[1,0:0,1]-[0,0]")
        s = format_pattern(canonicalize(p))
        self.assertEqual(s, "+[0,0:0,1]-[1,0]")

    def test_prompt_multi_stone_sequence_1(self):
        # g7,g8,h7,g9,f8  -> plus: g7,h7,f8 ; minus: g8,g9 ; anchored at g7
        p = parse_pattern("+[0,0:1,0:-1,1]-[0,1:0,2]")
        s = format_pattern(canonicalize(p))
        self.assertEqual(s, "+[0,0:0,1:1,-1]-[1,0:2,0]")

    def test_prompt_multi_stone_sequence_2(self):
        # g7,f9,h8,:p,g10 -> plus: g7,h8,g10 ; minus: f9 ; anchored at g7
        p = parse_pattern("+[0,0:1,1:0,3]-[-1,2]")
        s = format_pattern(canonicalize(p))
        self.assertEqual(s, "+[0,0:1,-2:1,1]-[2,-1]")

    def test_prompt_multi_stone_sequence_3(self):
        # g8,i7,:p,h9,i8,h8 -> plus: g8,i8 ; minus: i7,h9,h8 ; anchored at g8
        p = parse_pattern("+[0,0:2,0]-[2,-1:1,1:1,0]")
        s = format_pattern(canonicalize(p))
        self.assertEqual(s, "+[0,0:2,-2]-[1,-2:1,-1:2,-1]")


if __name__ == "__main__":
    unittest.main()

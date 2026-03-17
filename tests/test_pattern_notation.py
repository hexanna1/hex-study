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

    def test_canonical_output_examples(self):
        cases = (
            ("+[0,0]-[]", "+[0,0]-[]"),
            ("+[0,0:1,0]-[]", "+[0,0:0,1]-[]"),
            ("+[0,0]-[1,0]", "+[0,0]-[0,1]"),
            ("+[0,0:1,1]-[]", "+[0,0:1,-2]-[]"),
            ("+[0,0]-[1,1]", "+[0,0]-[1,-2]"),
            ("+[1,0:0,1]-[0,0]", "+[0,0:0,1]-[1,0]"),
            ("+[0,0:1,0:-1,1]-[0,1:0,2]", "+[0,0:0,1:1,-1]-[1,0:2,0]"),
            ("+[0,0:1,1:0,3]-[-1,2]", "+[0,0:1,-2:1,1]-[2,-1]"),
            ("+[0,0:2,0]-[2,-1:1,1:1,0]", "+[0,0:2,-2]-[1,-2:1,-1:2,-1]"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(format_pattern(canonicalize(parse_pattern(raw))), expected)


if __name__ == "__main__":
    unittest.main()

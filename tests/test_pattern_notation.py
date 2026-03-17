import unittest

from hex_symmetry import apply_transform_ax
from pattern_notation import LabeledPattern, UnlabeledPattern, canonicalize, format_pattern, parse_pattern


class PatternNotationTests(unittest.TestCase):
    def test_artifact_keys_are_stable_under_translation_symmetry_and_unlabeled_color_swap(self):
        # These keys identify saved artifacts; all equivalent geometric frames must agree.
        cases = (
            ("+[0,0:1,0:-1,1]-[0,1:0,2]", "+[0,0:0,1:1,-1]-[1,0:2,0]"),
            ("+[0,0:1,1:0,3]-[-1,2]", "+[0,0:1,-2:1,1]-[2,-1]"),
            ("+[0,0:2,0]-[2,-1:1,1:1,0]", "+[0,0:2,-2]-[1,-2:1,-1:2,-1]"),
        )
        for raw, expected in cases:
            pattern = parse_pattern(raw)
            unlabeled = canonicalize(UnlabeledPattern(pattern.plus, pattern.minus))
            for transform in range(12):
                def moved(points):
                    return tuple((q + 7, r - 13) for q, r in (
                        apply_transform_ax(point, transform) for point in reversed(points)
                    ))

                plus, minus = moved(pattern.plus), moved(pattern.minus)
                with self.subTest(pattern=raw, transform=transform):
                    self.assertEqual(format_pattern(canonicalize(LabeledPattern(plus, minus))), expected)
                    self.assertEqual(canonicalize(UnlabeledPattern(minus, plus)), unlabeled)

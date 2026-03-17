import unittest

import pattern_output_utils as sout


class PatternOutputUtilsTests(unittest.TestCase):
    def test_movelist_slug_from_hexworld(self):
        self.assertEqual(
            sout.movelist_slug_from_hexworld("https://hexworld.org/board/#25c1,m13:p"),
            "m13_p",
        )
        self.assertEqual(
            sout.movelist_slug_from_hexworld("https://hexworld.org/board/#25c1,m13n12n13o11"),
            "m13n12n13o11",
        )

if __name__ == "__main__":
    unittest.main()

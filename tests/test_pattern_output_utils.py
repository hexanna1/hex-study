import unittest

import pattern_output_utils as sout


class PatternOutputUtilsTests(unittest.TestCase):
    def test_local_map_title_color_uses_red_for_red_to_play(self):
        self.assertEqual(
            sout._local_map_title_color({"to_play": "red"}),
            (220 / 255.0, 60 / 255.0, 60 / 255.0),
        )

    def test_local_map_title_color_uses_blue_for_blue_to_play(self):
        self.assertEqual(
            sout._local_map_title_color({"to_play": "blue"}),
            (40 / 255.0, 100 / 255.0, 220 / 255.0),
        )

    def test_local_map_title_color_defaults_to_black(self):
        self.assertEqual(sout._local_map_title_color({}), "#111111")


if __name__ == "__main__":
    unittest.main()

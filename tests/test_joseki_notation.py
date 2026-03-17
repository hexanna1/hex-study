import unittest

from joseki_notation import (
    JosekiBlock,
    JosekiLine,
    format_joseki_line,
    format_single_track_line,
    parse_joseki_line,
)


class JosekiNotationTests(unittest.TestCase):
    def test_parse_and_format_multiple_blocks_with_empty_entries(self):
        raw = "A[5,4:3,7:]O[4,4]A[]"
        line = parse_joseki_line(raw)
        self.assertEqual(
            line,
            JosekiLine(
                blocks=(
                    JosekiBlock(family="A", entries=((5, 4), (3, 7), None)),
                    JosekiBlock(family="O", entries=((4, 4),)),
                    JosekiBlock(family="A", entries=(None,)),
                )
            ),
        )
        self.assertEqual(format_joseki_line(line), raw)
        self.assertEqual(format_single_track_line(family="o", entries=((4, 4), None, (2, 3))), "O[4,4::2,3]")

if __name__ == "__main__":
    unittest.main()

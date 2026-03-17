import unittest

from joseki_notation import (
    JosekiBlock,
    JosekiLine,
    format_joseki_line,
    format_single_track_line,
    merge_same_family_blocks,
    parse_joseki_line,
)


class JosekiNotationTests(unittest.TestCase):
    def test_parse_single_block_with_empty_entries(self):
        line = parse_joseki_line("A[5,4:3,7:]")
        self.assertEqual(
            line,
            JosekiLine(blocks=(JosekiBlock(family="A", entries=((5, 4), (3, 7), None)),)),
        )

    def test_parse_multiple_blocks(self):
        line = parse_joseki_line("A[5,4]O[4,4]A[]")
        self.assertEqual(len(line.blocks), 3)
        self.assertEqual(line.blocks[0].family, "A")
        self.assertEqual(line.blocks[1].family, "O")
        self.assertEqual(line.blocks[2].entries, (None,))

    def test_format_round_trip(self):
        raw = "A[5,4]O[4,4]A[]"
        self.assertEqual(format_joseki_line(parse_joseki_line(raw)), raw)

    def test_merge_same_family_blocks(self):
        line = parse_joseki_line("A[1,1]A[]A[2,2]O[4,4]")
        merged = merge_same_family_blocks(line)
        self.assertEqual(format_joseki_line(merged), "A[1,1::2,2]O[4,4]")

    def test_format_single_track_line(self):
        self.assertEqual(format_single_track_line(family="o", entries=((4, 4), None, (2, 3))), "O[4,4::2,3]")

    def test_parse_accepts_all_empty_line(self):
        self.assertEqual(
            parse_joseki_line("A[]"),
            JosekiLine(blocks=(JosekiBlock(family="A", entries=(None,)),)),
        )

    def test_format_single_track_line_accepts_all_empty_entries(self):
        self.assertEqual(format_single_track_line(family="A", entries=(None,)), "A[]")


if __name__ == "__main__":
    unittest.main()

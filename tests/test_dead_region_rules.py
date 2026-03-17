import unittest

import dead_region_rules as drr
import local_pattern_representative as lpr


def _points(*cells: str) -> set[tuple[int, int]]:
    return {tuple(int(v) for v in lpr.CELL_TO_COL_ROW(cell)) for cell in cells}


class DeadRegionRuleTests(unittest.TestCase):
    def test_canonicalize_acute_equivalent_move_prefers_c2_in_swapped_case(self):
        self.assertEqual(
            drr.canonicalize_acute_equivalent_move(
                move=tuple(int(v) for v in lpr.CELL_TO_COL_ROW("b3")),
                red=_points("c3"),
                blue=_points("d2", "d3"),
                board_size=19,
            ),
            tuple(int(v) for v in lpr.CELL_TO_COL_ROW("c2")),
        )


if __name__ == "__main__":
    unittest.main()

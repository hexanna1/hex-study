import unittest

import dead_region_rules as drr
import local_pattern_representative as lpr


def _points(*cells: str) -> set[tuple[int, int]]:
    return {tuple(int(v) for v in lpr.CELL_TO_COL_ROW(cell)) for cell in cells}


class DeadRegionRuleTests(unittest.TestCase):
    def test_acute_rule_context_prefers_c2_in_swapped_case(self):
        context = drr.acute_rule_context(
            red=_points("c3"),
            blue=_points("d2", "d3"),
            board_size=19,
        )
        self.assertEqual(
            drr.apply_acute_rule_context(
                move=tuple(int(v) for v in lpr.CELL_TO_COL_ROW("b3")),
                context=context,
            ),
            tuple(int(v) for v in lpr.CELL_TO_COL_ROW("c2")),
        )


if __name__ == "__main__":
    unittest.main()

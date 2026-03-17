import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pattern_enumeration as pattern_enumerate


class PatternEnumerationTests(unittest.TestCase):
    def test_single_stone_is_connected(self):
        self.assertTrue(pattern_enumerate._is_connected(((0, 0),), max_component_delta=4))

    def test_two_far_clusters_are_not_connected(self):
        pts = ((0, 0), (0, 1), (10, 0), (10, 1))
        self.assertFalse(pattern_enumerate._is_connected(pts, max_component_delta=4))

    def test_minimal_square_board_layout_uses_smallest_square_that_fits(self):
        board_size, mapped = pattern_enumerate._minimal_square_board_layout(((0, 0), (1, -2)))
        self.assertEqual(board_size, 3)
        self.assertEqual(set(mapped.values()), {(1, 3), (2, 1)})

    def test_enumerate_patterns_small_exact_catalog(self):
        catalog = pattern_enumerate.enumerate_patterns(max_moves=2, max_component_delta=4, study_delta=3)
        self.assertEqual(catalog["max_moves"], 2)
        self.assertEqual(catalog["total_patterns"], 5)
        self.assertEqual(
            catalog["counts_by_family"],
            [
                {"plus": 0, "minus": 1, "to_play": "blue", "count": 1},
                {"plus": 1, "minus": 0, "to_play": "red", "count": 1},
                {"plus": 1, "minus": 1, "to_play": "red", "count": 3},
            ],
        )
        patterns = [row["pattern"] for row in catalog["patterns"]]
        self.assertEqual(
            patterns,
            [
                "+[]-[0,0]",
                "+[0,0]-[]",
                "+[0,0]-[0,1]",
                "+[0,0]-[0,2]",
                "+[0,0]-[1,-2]",
            ],
        )
        by_pattern = {row["pattern"]: row for row in catalog["patterns"]}
        self.assertEqual(
            by_pattern["+[]-[0,0]"]["hexworld_21"],
            "https://hexworld.org/board/#21c1,k11",
        )
        self.assertEqual(by_pattern["+[]-[0,0]"]["to_play"], "blue")
        self.assertEqual(by_pattern["+[]-[0,0]"]["study_delta"], pattern_enumerate.DEFAULT_MAX_COMPONENT_DELTA)
        self.assertEqual(
            by_pattern["+[0,0]-[]"]["hexworld_21"],
            "https://hexworld.org/board/#21c1,k11:p",
        )
        self.assertEqual(by_pattern["+[0,0]-[]"]["to_play"], "red")
        self.assertEqual(by_pattern["+[0,0]-[]"]["study_delta"], pattern_enumerate.DEFAULT_MAX_COMPONENT_DELTA)
        self.assertEqual(
            by_pattern["+[0,0]-[0,1]"]["hexworld_21"],
            "https://hexworld.org/board/#21c1,k10k11",
        )
        self.assertEqual(by_pattern["+[0,0]-[0,1]"]["to_play"], "red")
        self.assertEqual(by_pattern["+[0,0]-[0,1]"]["study_delta"], 3)
    def test_max_pair_delta_filters_longer_two_stone_patterns(self):
        catalog = pattern_enumerate.enumerate_patterns(
            max_moves=2,
            max_component_delta=4,
            max_pair_delta=1,
        )
        self.assertEqual(catalog["total_patterns"], 3)
        patterns = [row["pattern"] for row in catalog["patterns"]]
        self.assertEqual(
            patterns,
            [
                "+[]-[0,0]",
                "+[0,0]-[]",
                "+[0,0]-[0,1]",
            ],
        )

    def test_enumerate_patterns_odd_max_moves_adds_only_the_families_within_budget(self):
        catalog = pattern_enumerate.enumerate_patterns(
            max_moves=3,
            max_component_delta=4,
        )
        self.assertEqual(catalog["max_moves"], 3)
        self.assertEqual(
            catalog["counts_by_family"],
            [
                {"plus": 0, "minus": 1, "to_play": "blue", "count": 1},
                {"plus": 1, "minus": 0, "to_play": "red", "count": 1},
                {"plus": 1, "minus": 1, "to_play": "red", "count": 3},
                {"plus": 0, "minus": 2, "to_play": "blue", "count": 3},
                {"plus": 1, "minus": 2, "to_play": "blue", "count": 26},
            ],
        )
        self.assertTrue(all((row["plus"] + row["minus"]) <= 3 for row in catalog["patterns"]))

    def test_canonical_families_respect_move_budget(self):
        self.assertEqual(
            pattern_enumerate._canonical_families(3),
            [
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (1, 2),
            ],
        )

    def test_to_play_for_labeled_family_is_red_for_ties_and_larger_plus(self):
        self.assertEqual(pattern_enumerate._to_play_for_labeled_family(1, 1), "red")
        self.assertEqual(pattern_enumerate._to_play_for_labeled_family(2, 1), "red")
        self.assertEqual(pattern_enumerate._to_play_for_labeled_family(1, 2), "blue")

    def test_moves_for_labeled_family_counts_one_tenuki_families_as_extra_move(self):
        self.assertEqual(pattern_enumerate._moves_for_labeled_family(1, 1), 2)
        self.assertEqual(pattern_enumerate._moves_for_labeled_family(1, 2), 3)
        self.assertEqual(pattern_enumerate._moves_for_labeled_family(2, 1), 4)
        self.assertEqual(pattern_enumerate._moves_for_labeled_family(1, 3), 5)
        self.assertEqual(pattern_enumerate._moves_for_labeled_family(2, 3), 5)
        self.assertEqual(pattern_enumerate._moves_for_labeled_family(3, 2), 6)

    def test_cross_color_closest_filter_only_rejects_delta1_same_color_when_cross_is_4_or_more(self):
        self.assertTrue(
            pattern_enumerate._passes_cross_color_closest_filter(
                red=((0, 0), (0, 4)),
                blue=((0, 1),),
            )
        )
        self.assertFalse(
            pattern_enumerate._passes_cross_color_closest_filter(
                red=((0, 0), (0, 1)),
                blue=((0, 4),),
            )
        )
        self.assertFalse(
            pattern_enumerate._passes_cross_color_closest_filter(
                red=((0, 0),),
                blue=((0, 3), (0, 4)),
            )
        )
        self.assertTrue(
            pattern_enumerate._passes_cross_color_closest_filter(
                red=((0, 0), (1, 1)),
                blue=((2, -2),),
            )
        )

    def test_write_catalog_png_pages_uses_shared_pagination_helper(self):
        catalog = {
            "max_moves": 2,
            "max_component_delta": 4,
            "max_pair_delta": 12,
            "total_patterns": 2,
            "patterns": [
                {"pattern": "+[]-[0,0]", "to_play": "blue"},
                {"pattern": "+[0,0]-[0,1]", "to_play": "red"},
            ],
        }
        with TemporaryDirectory() as td:
            out_dir = Path(td)
            with mock.patch(
                "pattern_enumeration.sout.write_local_map_contact_sheet_pages",
                return_value=["001.png"],
            ) as write_pages_mock:
                pages = pattern_enumerate.write_catalog_png_pages(catalog, out_dir)
        self.assertEqual(pages, ["001.png"])
        self.assertEqual(write_pages_mock.call_args.kwargs["columns"], pattern_enumerate.DEFAULT_PAGE_COLUMNS)
        self.assertEqual(write_pages_mock.call_args.kwargs["rows_per_page"], pattern_enumerate.DEFAULT_PAGE_ROWS)
        items = write_pages_mock.call_args.args[0]
        self.assertEqual([item["title"] for item in items], ["+[]-[0,0]", "+[0,0]-[0,1]"])
        self.assertEqual(items[0]["spec"]["to_play"], "blue")
        self.assertEqual(items[1]["spec"]["to_play"], "red")

    def test_main_can_run_study_batch_from_catalog(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            args = SimpleNamespace(
                max_moves=2,
                max_component_delta=4,
                max_pair_delta=12,
                out_dir=str(td_path / "patterns"),
                no_png=True,
                force_png=False,
                run_study=True,
                study_delta=3,
                study_workers=4,
                study_out_dir=None,
            )

            def fake_subprocess_run(cmd, cwd, start_new_session):
                self.assertEqual(cmd[0:2], ["python3", str(pattern_enumerate._repo_root() / "pattern_study_batch.py")])
                self.assertTrue(start_new_session)
                self.assertIn("--catalog", cmd)
                self.assertIn("--out-dir", cmd)
                self.assertIn("--no-png", cmd)
                study_out_dir = Path(cmd[cmd.index("--out-dir") + 1])
                study_out_dir.mkdir(parents=True, exist_ok=True)
                (study_out_dir / "manifest.json").write_text(
                    json.dumps({"png_pages": [], "png_page_count": 0}),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    pid=12345,
                    wait=lambda timeout=None: 0,
                    poll=lambda: 0,
                )

            with (
                mock.patch("pattern_enumeration._parse_args", return_value=args),
                mock.patch("pattern_enumeration.subprocess.Popen", side_effect=fake_subprocess_run),
                mock.patch("builtins.print") as print_mock,
            ):
                rc = pattern_enumerate.main()

        self.assertEqual(rc, 0)
        payload = json.loads(print_mock.call_args.args[0])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["study"]["out_dir"], str(Path(args.out_dir)))
        self.assertEqual(payload["study"]["manifest"]["png_pages"], [])

    def test_default_artifact_dir_uses_total_move_cap_label(self):
        path = pattern_enumerate._default_artifact_dir(
            max_moves=5,
            max_component_delta=7,
            max_pair_delta=12,
        )
        self.assertEqual(path, Path("artifacts") / "interior_patterns_m5_d7_span12")


if __name__ == "__main__":
    unittest.main()

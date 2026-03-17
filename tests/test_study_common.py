import unittest
from pathlib import Path
from unittest import mock

import study_common


class StudyCommonTests(unittest.TestCase):
    def test_run_once_empty_stdout_includes_stderr(self):
        fake_proc = mock.Mock(stdout="", stderr="python3: can't open file", returncode=2)
        with mock.patch("study_common.subprocess.run", return_value=fake_proc):
            ok, payload, err = study_common._run_once(["python3", "missing.py"])
        self.assertFalse(ok)
        self.assertIn("Empty stdout", payload["error"])
        self.assertEqual(payload["stderr"], "python3: can't open file")
        self.assertIn("stderr:", err)

    def test_build_cmd_omits_search_flags_when_budget_unspecified(self):
        fake_hexata_main = Path("fake_hexata_main.py")
        analyze_cmd = study_common._build_cmd(
            {"position": "#14c1,a1"},
            {},
            hexata_main=fake_hexata_main,
        )
        self.assertEqual(analyze_cmd[:4], ["python3", str(fake_hexata_main), "cli", "analyze"])
        self.assertNotIn("--search-seconds", analyze_cmd)
        self.assertNotIn("--top-n", analyze_cmd)
        self.assertNotIn("--awrn", analyze_cmd)

        candidate_cmd = study_common._build_cmd(
            {"position": "#14c1,a1", "candidates": ["d3", "e4"]},
            {},
            hexata_main=fake_hexata_main,
        )
        self.assertEqual(candidate_cmd[:4], ["python3", str(fake_hexata_main), "cli", "candidate"])
        self.assertNotIn("--search-seconds", candidate_cmd)

    def test_build_cmd_includes_search_flags_when_budget_provided(self):
        fake_hexata_main = Path("fake_hexata_main.py")
        analyze_cmd = study_common._build_cmd(
            {"position": "#14c1,a1", "search_seconds": 1.5, "awrn": 0.0},
            {},
            hexata_main=fake_hexata_main,
        )
        self.assertIn("--search-seconds", analyze_cmd)
        self.assertIn("1.5", analyze_cmd)
        self.assertIn("--awrn", analyze_cmd)

        candidate_cmd = study_common._build_cmd(
            {"position": "#14c1,a1", "candidates": ["d3"], "search_seconds": 2.0},
            {},
            hexata_main=fake_hexata_main,
        )
        self.assertIn("--search-seconds", candidate_cmd)
        self.assertIn("2.0", candidate_cmd)

    def test_aggregate_moves_computes_stone_fraction_from_proxy(self):
        payload = {
            "ok": True,
            "hexworld": "https://hexworld.org/board/#25c1",
            "candidate": {
                "method": "search",
                "moves": [
                    {"move": "m1", "red_winrate": 0.2, "visits": 10},
                    {"move": "j10", "red_winrate": 0.5, "visits": 10},
                    {"move": "q5", "red_winrate": 0.8, "visits": 10},
                ],
            },
        }
        rows = study_common._aggregate_moves(payload)
        by_move = {r["move"]: r for r in rows}
        self.assertAlmostEqual(by_move["m1"]["stone_fraction"], 0.0, places=6)
        self.assertAlmostEqual(by_move["q5"]["stone_fraction"], 1.0, places=6)
        self.assertGreater(by_move["j10"]["stone_fraction"], 0.0)
        self.assertLess(by_move["j10"]["stone_fraction"], 1.0)

    def test_aggregate_moves_uses_explicit_position_input_when_payload_omits_input(self):
        payload = {
            "ok": True,
            "hexworld": "https://hexworld.org/board/#25c1",
            "analyze": {
                "method": "search",
                "best": None,
                "moves": [
                    {"move": "m1", "red_winrate": 0.2, "visits": 10},
                    {"move": "j10", "red_winrate": 0.5, "visits": 10},
                    {"move": "q5", "red_winrate": 0.8, "visits": 10},
                ],
            },
        }
        rows = study_common._aggregate_moves(
            payload,
            position_input="https://hexworld.org/board/#25c1",
        )
        by_move = {r["move"]: r for r in rows}
        self.assertAlmostEqual(by_move["m1"]["stone_fraction"], 0.0, places=6)
        self.assertAlmostEqual(by_move["q5"]["stone_fraction"], 1.0, places=6)
        self.assertGreater(by_move["j10"]["stone_fraction"], 0.0)
        self.assertLess(by_move["j10"]["stone_fraction"], 1.0)

    def test_aggregate_moves_converts_red_winrate_for_blue_to_play(self):
        payload = {
            "ok": True,
            "hexworld": "https://hexworld.org/board/#25c1,a1",
            "candidate": {
                "method": "search",
                "moves": [
                    {"move": "a13", "red_winrate": 0.8, "visits": 10},  # proxy; blue wr = 0.2
                    {"move": "j10", "red_winrate": 0.5, "visits": 10},  # blue wr = 0.5
                    {"move": "q5", "red_winrate": 0.2, "visits": 10},   # blue wr = 0.8
                ],
            },
        }
        rows = study_common._aggregate_moves(payload)
        by_move = {r["move"]: r for r in rows}
        self.assertAlmostEqual(by_move["a13"]["mean_winrate"], 0.2, places=6)
        self.assertAlmostEqual(by_move["q5"]["mean_winrate"], 0.8, places=6)
        self.assertAlmostEqual(by_move["a13"]["stone_fraction"], 0.0, places=6)
        self.assertAlmostEqual(by_move["q5"]["stone_fraction"], 1.0, places=6)

    def test_candidate_key_local_roundtrip_for_multiple_transforms(self):
        base_rel = (2, -1)
        offset = (10, 10)
        for transform_id in (0, 1, 7):
            transformed = study_common._apply_transform_ax(base_rel, transform_id)
            abs_col = offset[0] + transformed[0]
            abs_row = offset[1] + transformed[1]
            move = f"{study_common._letters_for_col(abs_col)}{abs_row}"
            exp_meta = {
                "orientation_transform_id": transform_id,
                "orientation_norm_shift": [0, 0],
                "placement_offset": [offset[0], offset[1]],
            }
            key = study_common._candidate_key_local_for_move(move, exp_meta)
            self.assertEqual(key, f"{base_rel[0]},{base_rel[1]}")

    def test_candidate_key_local_orbit_canonicalizes_symmetric_geometric_keys(self):
        exp_meta_base = {
            "orientation_transform_id": 1,
            "orientation_norm_shift": [0, 0],
            "placement_offset": [10, 10],
        }

        def move_for_base_rel(base_rel: tuple[int, int], exp_meta: dict) -> str:
            transform_id = int(exp_meta["orientation_transform_id"])
            shift = exp_meta["orientation_norm_shift"]
            offset = exp_meta["placement_offset"]
            transformed = study_common._apply_transform_ax(base_rel, transform_id)
            col = int(offset[0]) + transformed[0] - int(shift[0])
            row = int(offset[1]) + transformed[1] - int(shift[1])
            return f"{study_common._letters_for_col(col)}{row}"

        move_a = move_for_base_rel((0, 1), exp_meta_base)
        move_b = move_for_base_rel((1, -1), exp_meta_base)

        key_a_raw = study_common._candidate_key_local_for_move(move_a, exp_meta_base)
        key_b_raw = study_common._candidate_key_local_for_move(move_b, exp_meta_base)
        self.assertEqual(key_a_raw, "0,1")
        self.assertEqual(key_b_raw, "1,-1")

        exp_meta_orbit = {
            **exp_meta_base,
            "local_key_orbit": [
                {"transform_id": 0, "norm_shift": [0, 0]},
                {"transform_id": 3, "norm_shift": [-1, 0]},
                {"transform_id": 8, "norm_shift": [-1, 0]},
                {"transform_id": 11, "norm_shift": [0, 0]},
            ],
        }
        key_a = study_common._candidate_key_local_for_move(move_a, exp_meta_orbit)
        key_b = study_common._candidate_key_local_for_move(move_b, exp_meta_orbit)
        self.assertEqual(key_a, "0,1")
        self.assertEqual(key_b, "0,1")

    def test_build_pooled_candidates_groups_fraction_rows_by_local_key(self):
        rows = [
            {
                "experiment": "e1",
                "candidate_key_local": "1,0",
                "candidate_abs": "m10",
                "stone_fraction": 0.8,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "1,0",
                "candidate_abs": "n9",
                "stone_fraction": 0.6,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "2,0",
                "candidate_abs": "n10",
                "stone_fraction": 0.3,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "2,0",
                "candidate_abs": "o9",
                "stone_fraction": 0.5,
            },
            {
                "experiment": "e2",
                "candidate_key_local": None,
                "candidate_abs": "junk",
                "stone_fraction": 0.9,
            },
        ]
        pooled = study_common._build_pooled_candidates(
            rows,
            total_representatives=2,
            value_field="stone_fraction",
        )
        self.assertEqual(len(pooled), 2)
        by_key = {r["candidate_key_local"]: r for r in pooled}
        self.assertAlmostEqual(by_key["1,0"]["mean_stone_fraction"], 0.7, places=6)
        self.assertAlmostEqual(by_key["2,0"]["mean_stone_fraction"], 0.4, places=6)
        self.assertEqual(by_key["1,0"]["n"], 2)
        self.assertAlmostEqual(by_key["1,0"]["coverage"], 1.0, places=6)

    def test_build_pooled_candidates_normalizes_after_pooling(self):
        rows = [
            {
                "experiment": "e1",
                "candidate_key_local": "1,0",
                "candidate_abs": "m10",
                "corrected_value": 0.8,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "1,0",
                "candidate_abs": "n9",
                "corrected_value": 0.6,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "2,0",
                "candidate_abs": "n10",
                "corrected_value": 0.3,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "2,0",
                "candidate_abs": "o9",
                "corrected_value": 0.5,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "pass_proxy",
                "candidate_abs": "m1",
                "corrected_value": 0.0,
            },
        ]
        pooled = study_common._build_pooled_candidates(rows, total_representatives=2)
        self.assertEqual(len(pooled), 3)
        by_key = {r["candidate_key_local"]: r for r in pooled}
        self.assertAlmostEqual(by_key["1,0"]["mean_corrected_value"], 0.7, places=6)
        self.assertAlmostEqual(by_key["2,0"]["mean_corrected_value"], 0.4, places=6)
        self.assertAlmostEqual(by_key["1,0"]["mean_stone_fraction"], 1.0, places=6)
        self.assertAlmostEqual(by_key["2,0"]["mean_stone_fraction"], 4.0 / 7.0, places=6)
        self.assertAlmostEqual(by_key["pass_proxy"]["mean_stone_fraction"], 0.0, places=6)

    def test_build_pooled_candidates_all_zero_when_pass_is_best_on_average(self):
        rows = [
            {
                "experiment": "e1",
                "candidate_key_local": "1,0",
                "candidate_abs": "m10",
                "corrected_value": -0.1,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "1,0",
                "candidate_abs": "n9",
                "corrected_value": -0.2,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "pass_proxy",
                "candidate_abs": "m1",
                "corrected_value": 0.0,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "pass_proxy",
                "candidate_abs": "m1",
                "corrected_value": 0.0,
            },
        ]
        pooled = study_common._build_pooled_candidates(
            rows,
            total_representatives=2,
            value_field="corrected_value",
        )
        by_key = {r["candidate_key_local"]: r for r in pooled}
        self.assertAlmostEqual(by_key["pass_proxy"]["mean_corrected_value"], 0.0, places=6)
        self.assertAlmostEqual(by_key["pass_proxy"]["mean_stone_fraction"], 0.0, places=6)
        self.assertAlmostEqual(by_key["1,0"]["mean_stone_fraction"], 0.0, places=6)

if __name__ == "__main__":
    unittest.main()

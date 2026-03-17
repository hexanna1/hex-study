import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pattern_study
import study_common as lps


class PatternStudyTests(unittest.TestCase):
    def test_cache_key_canonicalizes_same_position_across_same_color_order(self):
        self.assertEqual(
            lps._cache_key("https://hexworld.org/board/#21c1,a1d2k11l11"),
            lps._cache_key("https://hexworld.org/board/#21c1,k11d2a1l11"),
        )

    def test_cache_child_key_uses_short_canonical_position_form(self):
        self.assertEqual(
            lps._cache_key("https://hexworld.org/board/#21c1,a1d2k11l11", "l12"),
            "21,a1d2k11l11l12",
        )

    def test_raw_nn_cache_round_trips_via_json_file(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "raw_nn_cache.json"
            pattern_study._save_raw_nn_cache(path, {"21,a1d2": 0.5, "21,a1d2k11l11l12": 0.6})
            loaded = pattern_study._load_raw_nn_cache(path)
        self.assertEqual(loaded, {"21,a1d2": 0.5, "21,a1d2k11l11l12": 0.6})

    def test_fmt_s_rounds_before_switching_to_minute_format(self):
        self.assertEqual(lps._fmt_s(59.95), "1m")

    def test_movelist_slug_from_hexworld(self):
        self.assertEqual(
            pattern_study.sout._movelist_slug_from_hexworld("https://hexworld.org/board/#25c1,m13:p"),
            "m13_p",
        )
        self.assertEqual(
            pattern_study.sout._movelist_slug_from_hexworld("https://hexworld.org/board/#25c1,m13n12n13o11"),
            "m13n12n13o11",
        )

    def test_fmt_s_includes_hours(self):
        self.assertEqual(lps._fmt_s(3665.0), "1h1m05s")

    def test_select_root_tenuki_move_uses_ranked_outside_move(self):
        payload = {
            "moves": [
                {"move": "m13", "rank": 1, "prior": 0.9},
                {"move": "k13", "rank": 2, "prior": 0.5},
                {"move": "l13", "rank": 3, "prior": 0.7},
            ]
        }
        tenuki = pattern_study._select_root_tenuki_move(
            payload,
            {"m13"},
            pattern_cells_abs={(13, 13)},
            tenuki_Δ_min=0,
        )
        self.assertEqual(tenuki, "k13")

    def test_select_root_tenuki_move_ignores_non_cell_tokens(self):
        payload = {
            "moves": [
                {"move": "pass", "rank": 1, "prior": 1.0},
                {"move": "m13", "rank": 2, "prior": 0.9},
                {"move": "l13", "rank": 3, "prior": 0.8},
            ]
        }
        tenuki = pattern_study._select_root_tenuki_move(
            payload,
            {"m13"},
            pattern_cells_abs={(13, 13)},
            tenuki_Δ_min=0,
        )
        self.assertEqual(tenuki, "l13")

    def test_select_root_tenuki_move_applies_min_Δ_gate(self):
        payload = {
            "moves": [
                {"move": "n13", "rank": 1, "prior": 0.9},  # Δ=1 from m13, excluded
                {"move": "e13", "rank": 2, "prior": 0.8},  # Δ=64 from m13, eligible
            ]
        }
        tenuki = pattern_study._select_root_tenuki_move(
            payload,
            {"m13"},
            pattern_cells_abs={(13, 13)},
            tenuki_Δ_min=37,
        )
        self.assertEqual(tenuki, "e13")

    def test_select_root_tenuki_move_returns_none_when_no_move_meets_min_Δ(self):
        payload = {
            "moves": [
                {"move": "n13", "rank": 1, "prior": 0.9},
                {"move": "l13", "rank": 2, "prior": 0.8},
            ]
        }
        tenuki = pattern_study._select_root_tenuki_move(
            payload,
            {"m13"},
            pattern_cells_abs={(13, 13)},
            tenuki_Δ_min=37,
        )
        self.assertIsNone(tenuki)

    def test_main_canonical_blue_reuses_shared_without_root_probe(self):
        rep = pattern_study.Representative(
            balance_index=1,
            orientation_index=1,
            orientation_transform_id=0,
            orientation_norm_shift=(0, 0),
            placement_offset=(0, 0),
            balance_moves=("a1", "d2"),
            to_play_at_cursor="blue",
            plus_abs=((10, 10),),
            minus_abs=((11, 11),),
            red_cells=((1, 1), (11, 11)),
            blue_cells=((2, 2), (10, 10)),
            position="https://hexworld.org/board/#25c1,a1d2",
        )
        base_spec = {
            "experiments": [
                {"name": "b01-o01", "label": "e1", "position": "https://hexworld.org/board/#25c1,a2", "candidates": ["a2"]},
                {"name": "b01-o02", "label": "e2", "position": "https://hexworld.org/board/#25c1,a1", "candidates": ["a1"]},
            ],
            "generator_meta": {"candidate_Δ_max": 7, "experiment_meta": {}},
        }
        args = SimpleNamespace(
            hexworld="https://hexworld.org/board/#25c1,a1",
            candidate_Δ_max=7,
            board_size=25,
            symmetry="d6",
            placement="centered",
            edge_anchor_col_from_right=1,
            tenuki_Δ_min=21,
            out_dir="",
        )
        extracted = SimpleNamespace(
            plus_rel=((0, 0),),
            minus_rel=((1, 1),),
            to_play_at_cursor="blue",
            board_size_source=25,
        )
        probe_payload = {
            "moves": [
                {"move": "a1", "rank": 1, "prior": 1.0},
                {"move": "d2", "rank": 2, "prior": 0.9},
                {"move": "e13", "rank": 3, "prior": 0.8},
            ]
        }
        root_payload = {"root_eval": {"red_winrate": 0.5}, "position": {"to_play": "blue"}}
        run_once_mock = mock.Mock(side_effect=[
            (True, probe_payload, ""),
        ])

        with TemporaryDirectory() as td:
            hexata_main = Path(td) / "main.py"
            cache_path = Path(td) / "raw_nn_cache.json"
            hexata_main.write_text("", encoding="utf-8")
            args.out_dir = str(Path(td) / "out")
            with (
                mock.patch("pattern_study._parse_args", return_value=args),
                mock.patch("pattern_study.extract_pattern", return_value=extracted),
                mock.patch("pattern_study.parse_balance_profiles", return_value=[]),
                mock.patch("pattern_study.generate_representatives", return_value=[rep, rep]),
                mock.patch("pattern_study.build_study_spec", return_value=base_spec),
                mock.patch("pattern_study.lps._hexata_main_path", return_value=hexata_main),
                mock.patch("pattern_study._raw_nn_cache_path", return_value=cache_path),
                mock.patch("pattern_study.lps._run_once", run_once_mock),
                mock.patch(
                    "pattern_study.lps._run_multi_position_analyze",
                    return_value={
                        "https://hexworld.org/board/#25c1,a2": root_payload,
                        "https://hexworld.org/board/#25c1,a1": root_payload,
                        "https://hexworld.org/board/#25c1,a1d2e13": root_payload,
                    },
                ),
                mock.patch("pattern_study._select_root_tenuki_move", return_value=None),
                mock.patch("pattern_study.lps._log"),
            ):
                with self.assertRaisesRegex(ValueError, "Pass-proxy candidate was removed by occupancy filtering"):
                    pattern_study.main()

        self.assertEqual(run_once_mock.call_count, 1)

    def test_main_edge_defaults_symmetry_to_edge_bilateral(self):
        args = SimpleNamespace(
            hexworld="https://hexworld.org/board/#25c1,m13",
            candidate_Δ_max=7,
            board_size=25,
            symmetry=None,
            placement="edge",
            edge_anchor_col_from_right=5,
            tenuki_Δ_min=21,
            out_dir="",
            debug=False,
        )
        extracted = SimpleNamespace(
            plus_rel=(),
            minus_rel=((0, 0),),
            plus_cells=(),
            minus_cells=("m13",),
            to_play_at_cursor="blue",
            board_size_source=25,
        )
        parse_balance_mock = mock.Mock(return_value=pattern_study.parse_balance_profiles(["a1,d2"]))
        generate_reps_mock = mock.Mock(side_effect=RuntimeError("stop"))
        with (
            mock.patch("pattern_study._parse_args", return_value=args),
            mock.patch("pattern_study.extract_pattern", return_value=extracted),
            mock.patch("pattern_study.parse_balance_profiles", parse_balance_mock),
            mock.patch("pattern_study.generate_representatives", generate_reps_mock),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                pattern_study.main()
        parse_balance_mock.assert_called_once_with(["a1,d2"])
        generate_reps_mock.assert_called_once_with(
            extracted=extracted,
            board_size=25,
            symmetry="edge-bilateral",
            balance_profiles=parse_balance_mock.return_value,
            placement="edge",
            edge_anchor_col_from_right=5,
        )

    def test_main_nondebug_writes_compact_local_map_json_only(self):
        rep = pattern_study.Representative(
            balance_index=1,
            orientation_index=1,
            orientation_transform_id=0,
            orientation_norm_shift=(0, 0),
            placement_offset=(0, 0),
            balance_moves=("a1", "d2"),
            to_play_at_cursor="red",
            plus_abs=((10, 10),),
            minus_abs=((11, 11),),
            red_cells=((1, 1), (10, 10)),
            blue_cells=((4, 2), (11, 11)),
            position="https://hexworld.org/board/#21c1,a1d2j10k11",
        )
        base_spec = {
            "experiments": [
                {
                    "name": "b01-o01",
                    "label": "e1",
                    "position": rep.position,
                    "candidates": ["k1", "j11"],
                }
            ],
            "generator_meta": {
                "candidate_Δ_max": 7,
                "experiment_meta": {
                    "b01-o01": {
                        "orientation_transform_id": 0,
                        "orientation_norm_shift": [0, 0],
                        "placement_offset": [10, 10],
                    }
                },
            },
        }
        args = SimpleNamespace(
            hexworld="https://hexworld.org/board/#21c1,k11",
            candidate_Δ_max=7,
            board_size=21,
            symmetry="d6",
            placement="centered",
            edge_anchor_col_from_right=1,
            tenuki_Δ_min=21,
            out_dir="",
            debug=False,
        )
        extracted = SimpleNamespace(
            plus_rel=((0, 0),),
            minus_rel=((1, 1),),
            to_play_at_cursor="red",
            board_size_source=21,
        )
        root_payload = {
            "root_eval": {"red_winrate": 0.5},
            "position": {"to_play": "red"},
            "mode": "analyze",
            "moves": [{"move": "j11", "red_winrate": 0.6}],
        }
        write_json_mock = mock.Mock()
        child_payloads = {
            lps._position_after_move(rep.position, "k1"): {"root_eval": {"red_winrate": 0.4}},
            lps._position_after_move(rep.position, "j11"): {"root_eval": {"red_winrate": 0.6}},
            lps._position_after_move("https://hexworld.org/board/#21c1,a1d2", "k1"): {"root_eval": {"red_winrate": 0.4}},
            lps._position_after_move("https://hexworld.org/board/#21c1,a1d2", "j11"): {"root_eval": {"red_winrate": 0.6}},
        }

        with TemporaryDirectory() as td:
            hexata_main = Path(td) / "main.py"
            cache_path = Path(td) / "raw_nn_cache.json"
            hexata_main.write_text("", encoding="utf-8")
            with (
                mock.patch("pattern_study._parse_args", return_value=args),
                mock.patch("pattern_study.extract_pattern", return_value=extracted),
                mock.patch("pattern_study.parse_balance_profiles", return_value=pattern_study.parse_balance_profiles(["a1,d2"])),
                mock.patch("pattern_study.generate_representatives", return_value=[rep]),
                mock.patch("pattern_study.build_study_spec", return_value=base_spec),
                mock.patch("pattern_study.lps._hexata_main_path", return_value=hexata_main),
                mock.patch("pattern_study._raw_nn_cache_path", return_value=cache_path),
                mock.patch(
                    "pattern_study.lps._run_multi_position_analyze",
                    side_effect=[
                        {
                            rep.position: root_payload,
                            "https://hexworld.org/board/#21c1,a1d2": root_payload,
                        },
                        child_payloads,
                    ],
                ),
                mock.patch("pattern_study._select_root_tenuki_move", return_value=None),
                mock.patch("pattern_study.sout._write_local_map_spec_json", write_json_mock),
                mock.patch("pattern_study.lps._log"),
            ):
                rc = pattern_study.main()

        self.assertEqual(rc, 0)
        write_json_mock.assert_called_once()
        out_path = write_json_mock.call_args.args[0]
        spec = write_json_mock.call_args.args[1]
        self.assertEqual(Path(out_path).suffix, ".json")
        self.assertEqual(set(spec.keys()), {"pattern", "to_play", "cells"})
        self.assertEqual(spec["pattern"], "+[0,0]-[1,-2]")
        self.assertEqual(spec["to_play"], "red")
        self.assertEqual(
            spec["cells"],
            [
                {"kind": "local", "key": "0,1", "local_rel": [1, -1], "stone_fraction": 1.0, "rank": 1},
            ],
        )
    def test_main_nondebug_respects_out_dir_for_json_destination(self):
        rep = pattern_study.Representative(
            balance_index=1,
            orientation_index=1,
            orientation_transform_id=0,
            orientation_norm_shift=(0, 0),
            placement_offset=(0, 0),
            balance_moves=("a1", "d2"),
            to_play_at_cursor="red",
            plus_abs=((10, 10),),
            minus_abs=((11, 11),),
            red_cells=((1, 1), (10, 10)),
            blue_cells=((4, 2), (11, 11)),
            position="https://hexworld.org/board/#21c1,a1d2j10k11",
        )
        base_spec = {
            "experiments": [
                {
                    "name": "b01-o01",
                    "label": "e1",
                    "position": rep.position,
                    "candidates": ["k1", "j11"],
                }
            ],
            "generator_meta": {
                "candidate_Δ_max": 7,
                "experiment_meta": {
                    "b01-o01": {
                        "orientation_transform_id": 0,
                        "orientation_norm_shift": [0, 0],
                        "placement_offset": [10, 10],
                    }
                },
            },
        }
        root_payload = {
            "root_eval": {"red_winrate": 0.5},
            "position": {"to_play": "red"},
            "mode": "analyze",
            "moves": [{"move": "j11", "red_winrate": 0.6}],
        }
        extracted = SimpleNamespace(
            plus_rel=((0, 0),),
            minus_rel=((1, 1),),
            to_play_at_cursor="red",
            board_size_source=21,
        )
        child_payloads = {
            lps._position_after_move(rep.position, "k1"): {"root_eval": {"red_winrate": 0.4}},
            lps._position_after_move(rep.position, "j11"): {"root_eval": {"red_winrate": 0.6}},
            lps._position_after_move("https://hexworld.org/board/#21c1,a1d2", "k1"): {"root_eval": {"red_winrate": 0.4}},
            lps._position_after_move("https://hexworld.org/board/#21c1,a1d2", "j11"): {"root_eval": {"red_winrate": 0.6}},
        }

        with TemporaryDirectory() as td:
            hexata_main = Path(td) / "main.py"
            cache_path = Path(td) / "raw_nn_cache.json"
            hexata_main.write_text("", encoding="utf-8")
            custom_out_dir = Path(td) / "custom-json-dir"
            args = SimpleNamespace(
                hexworld="https://hexworld.org/board/#21c1,k11",
                candidate_Δ_max=7,
                board_size=21,
                symmetry="d6",
                placement="centered",
                edge_anchor_col_from_right=1,
                tenuki_Δ_min=21,
                out_dir=str(custom_out_dir),
                debug=False,
            )
            write_json_mock = mock.Mock()
            with (
                mock.patch("pattern_study._parse_args", return_value=args),
                mock.patch("pattern_study.extract_pattern", return_value=extracted),
                mock.patch("pattern_study.parse_balance_profiles", return_value=pattern_study.parse_balance_profiles(["a1,d2"])),
                mock.patch("pattern_study.generate_representatives", return_value=[rep]),
                mock.patch("pattern_study.build_study_spec", return_value=base_spec),
                mock.patch("pattern_study.lps._hexata_main_path", return_value=hexata_main),
                mock.patch("pattern_study._raw_nn_cache_path", return_value=cache_path),
                mock.patch(
                    "pattern_study.lps._run_multi_position_analyze",
                    side_effect=[
                        {
                            rep.position: root_payload,
                            "https://hexworld.org/board/#21c1,a1d2": root_payload,
                        },
                        child_payloads,
                    ],
                ),
                mock.patch("pattern_study._select_root_tenuki_move", return_value=None),
                mock.patch("pattern_study.sout._write_local_map_spec_json", write_json_mock),
                mock.patch("pattern_study.lps._log"),
            ):
                rc = pattern_study.main()

        self.assertEqual(rc, 0)
        out_path = Path(write_json_mock.call_args.args[0])
        self.assertEqual(out_path.parent.resolve(), custom_out_dir.resolve())
        self.assertEqual(out_path.suffix, ".json")

    def test_apply_ablation_calibration_anchors_to_pass_and_local_best(self):
        rows_with = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.40,
            },
            {
                "experiment": "e1",
                "move": "m13",
                "candidate_abs": "m13",
                "candidate_key_local": "0,0",
                "stone_fraction": 1.0,
                "mean_winrate": 0.60,
            },
            {
                "experiment": "e1",
                "move": "l13",
                "candidate_abs": "l13",
                "candidate_key_local": "-1,0",
                "stone_fraction": 0.75,
                "mean_winrate": 0.55,
            },
            {
                "experiment": "e1",
                "move": "z99",
                "candidate_abs": "z99",
                "candidate_key_local": "x",
                "stone_fraction": 0.0,
                "mean_winrate": None,
            },
        ]
        rows_without = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.38,
            },
            {
                "experiment": "e1",
                "move": "m13",
                "candidate_abs": "m13",
                "candidate_key_local": "0,0",
                "stone_fraction": 1.0,
                "mean_winrate": 0.56,
            },
            {
                "experiment": "e1",
                "move": "l13",
                "candidate_abs": "l13",
                "candidate_key_local": "-1,0",
                "stone_fraction": 0.75,
                "mean_winrate": 0.49,
            },
        ]
        root_ablation = {"e1": {"with_root_winrate": 0.52, "without_root_winrate": 0.50}}
        with mock.patch("pattern_study.lps._log"):
            out, diag = pattern_study._apply_ablation_calibration_to_summary_rows(
                summary_rows_with=rows_with,
                summary_rows_without=rows_without,
                root_ablation_by_experiment=root_ablation,
            )
        self.assertEqual(len(out), 3)
        by_move = {str(r.get("move")): r for r in out}
        self.assertAlmostEqual(by_move["m1"]["corrected_value"], 0.0, places=6)
        self.assertAlmostEqual(by_move["m13"]["stone_fraction_pre_ablation"], 1.0, places=6)
        self.assertIsInstance(by_move["m13"]["corrected_value"], float)
        self.assertGreater(by_move["m13"]["corrected_value"], 0.0)
        self.assertIsInstance(by_move["l13"]["corrected_value"], float)
        self.assertGreater(by_move["l13"]["corrected_value"], by_move["m13"]["corrected_value"])
        self.assertAlmostEqual(by_move["l13"]["stone_fraction_pre_ablation"], 0.75, places=6)
        self.assertAlmostEqual(by_move["l13"]["mean_winrate_without_pattern"], 0.49, places=6)
        self.assertIsInstance(by_move["l13"]["ablation_interaction_lift_logit"], float)
        self.assertAlmostEqual(rows_with[1]["stone_fraction"], 1.0, places=6)  # input rows unchanged
        self.assertNotIn("stone_fraction", by_move["l13"])
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag[0]["numeric_row_count"], 3)
        self.assertEqual(diag[0]["paired_row_count"], 3)
        self.assertEqual(diag[0]["anchor_pass_move"], "m1")
        self.assertEqual(diag[0]["anchor_best_local_move"], "l13")
        self.assertGreater(float(diag[0]["max_corrected_value"]), 0.0)

    def test_apply_ablation_calibration_coalesces_orbit_keys_before_scoring(self):
        rows_with = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.40,
                "n": 1,
            },
            {
                "experiment": "e1",
                "move": "m13",
                "candidate_abs": "m13",
                "candidate_key_local": "0,0",
                "stone_fraction": 0.90,
                "mean_winrate": 0.62,
                "n": 1,
            },
            {
                "experiment": "e1",
                "move": "n13",
                "candidate_abs": "n13",
                "candidate_key_local": "0,0",
                "stone_fraction": 0.80,
                "mean_winrate": 0.58,
                "n": 1,
            },
            {
                "experiment": "e1",
                "move": "l13",
                "candidate_abs": "l13",
                "candidate_key_local": "-1,0",
                "stone_fraction": 0.70,
                "mean_winrate": 0.55,
                "n": 1,
            },
        ]
        rows_without = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.38,
                "n": 1,
            },
            {
                "experiment": "e1",
                "move": "m13",
                "candidate_abs": "m13",
                "candidate_key_local": "0,0",
                "stone_fraction": 0.85,
                "mean_winrate": 0.56,
                "n": 1,
            },
            {
                "experiment": "e1",
                "move": "n13",
                "candidate_abs": "n13",
                "candidate_key_local": "0,0",
                "stone_fraction": 0.75,
                "mean_winrate": 0.54,
                "n": 1,
            },
            {
                "experiment": "e1",
                "move": "l13",
                "candidate_abs": "l13",
                "candidate_key_local": "-1,0",
                "stone_fraction": 0.65,
                "mean_winrate": 0.49,
                "n": 1,
            },
        ]
        root_ablation = {"e1": {"with_root_winrate": 0.52, "without_root_winrate": 0.50}}
        with mock.patch("pattern_study.lps._log"):
            out, diag = pattern_study._apply_ablation_calibration_to_summary_rows(
                summary_rows_with=rows_with,
                summary_rows_without=rows_without,
                root_ablation_by_experiment=root_ablation,
            )
        self.assertEqual(len(out), 3)
        by_key = {str(r.get("candidate_key_local")): r for r in out}
        self.assertIn("0,0", by_key)
        self.assertEqual(by_key["0,0"]["n"], 2)
        self.assertAlmostEqual(by_key["0,0"]["mean_winrate"], 0.60, places=6)
        self.assertAlmostEqual(by_key["0,0"]["mean_winrate_without_pattern"], 0.55, places=6)
        self.assertIsInstance(by_key["0,0"]["corrected_value"], float)
        self.assertEqual(diag[0]["paired_row_count"], 3)
        self.assertEqual(diag[0]["anchor_pass_move"], "m1")

    def test_apply_ablation_calibration_fails_if_numeric_with_row_lacks_paired_without_eval(self):
        rows_with = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.40,
            },
            {
                "experiment": "e1",
                "move": "m13",
                "candidate_abs": "m13",
                "candidate_key_local": "0,0",
                "stone_fraction": 1.0,
                "mean_winrate": 0.60,
            },
        ]
        rows_without = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.38,
            },
            {
                "experiment": "e1",
                "move": "m13",
                "candidate_abs": "m13",
                "candidate_key_local": "0,0",
                "stone_fraction": 1.0,
                "mean_winrate": None,
            },
        ]
        root_ablation = {"e1": {"with_root_winrate": 0.52, "without_root_winrate": 0.50}}
        with self.assertRaisesRegex(ValueError, "missing paired without-pattern eval"):
            pattern_study._apply_ablation_calibration_to_summary_rows(
                summary_rows_with=rows_with,
                summary_rows_without=rows_without,
                root_ablation_by_experiment=root_ablation,
            )

    def test_apply_ablation_calibration_restores_pass_class_term_via_ablation_pass_proxy(self):
        rows_with = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.25,
            },
            {
                "experiment": "e1",
                "move": "a1",
                "candidate_abs": "a1",
                "candidate_key_local": "a",
                "stone_fraction": 0.5,
                "mean_winrate": 0.60,
            },
            {
                "experiment": "e1",
                "move": "b1",
                "candidate_abs": "b1",
                "candidate_key_local": "b",
                "stone_fraction": 1.0,
                "mean_winrate": 0.70,
            },
        ]
        rows_without = [
            {
                "experiment": "e1",
                "move": "m1",
                "candidate_abs": "m1",
                "candidate_key_local": pattern_study.PASS_PROXY_CANONICAL_KEY,
                "stone_fraction": 0.0,
                "mean_winrate": 0.20,
            },
            {
                "experiment": "e1",
                "move": "a1",
                "candidate_abs": "a1",
                "candidate_key_local": "a",
                "stone_fraction": 0.5,
                "mean_winrate": 0.55,
            },
            {
                "experiment": "e1",
                "move": "b1",
                "candidate_abs": "b1",
                "candidate_key_local": "b",
                "stone_fraction": 1.0,
                "mean_winrate": 0.55,
            },
        ]
        root_ablation = {"e1": {"with_root_winrate": 0.50, "without_root_winrate": 0.50}}
        with mock.patch("pattern_study.lps._log"):
            out, diag = pattern_study._apply_ablation_calibration_to_summary_rows(
                summary_rows_with=rows_with,
                summary_rows_without=rows_without,
                root_ablation_by_experiment=root_ablation,
            )

        by_move = {str(r.get("move")): r for r in out}
        # Under pure interaction anchoring, a1 would be negative because I(a1) < I(pass_proxy).
        # The pass-class-restored score should still place a real move above the pass floor.
        self.assertLess(
            float(by_move["a1"]["ablation_interaction_lift_logit"]),
            float(by_move["m1"]["ablation_interaction_lift_logit"]),
        )
        self.assertGreater(float(by_move["a1"]["corrected_value"]), 0.0)
        self.assertAlmostEqual(float(by_move["m1"]["corrected_value"]), 0.0, places=6)
        self.assertGreater(float(by_move["b1"]["corrected_value"]), float(by_move["a1"]["corrected_value"]))
        self.assertGreater(float(diag[0]["generic_move_value_logit_hat"]), 0.0)


if __name__ == "__main__":
    unittest.main()

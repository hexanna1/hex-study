import unittest
from pathlib import Path
from unittest import mock

import joseki_database as jd

PARENT_POS = "https://hexworld.org/board/#19c1"


class JosekiDatabaseTests(unittest.TestCase):
    def test_cached_multi_position_analyze_reuses_cached_payloads(self):
        pos1 = "https://hexworld.org/board/#19c1,a1d2"
        pos2 = "https://hexworld.org/board/#19c1,a1d2p15"
        cache = {jd.lps._cache_key(pos1): {"r": 0.5, "m": [["p15", 0.2]]}}
        with mock.patch(
            "joseki_database.lps._run_multi_position_analyze",
            return_value={
                pos2: {
                    "ok": True,
                    "hexworld": pos2,
                    "analyze": {
                        "method": "raw_nn",
                        "best": None,
                        "root_eval": {"red_winrate": 0.6},
                        "moves": [{"move": "q14", "prior": 0.1}],
                    },
                    "meta": {"elapsed_ms": 1},
                }
            },
        ) as mocked:
            payloads = jd._run_multi_position_analyze_cached(
                hexata_main=Path("dummy"),
                position_inputs=[pos1, pos2, pos2],
                raw_nn_cache=cache,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(payloads[pos1]["r"], 0.5)
        self.assertEqual(payloads[pos2]["r"], 0.6)
        self.assertEqual(payloads[pos2]["m"], [["q14", 0.1]])

    def test_root_ply_uses_configured_local_moves_and_no_tenuki(self):
        red, blue = jd._balance_cells(family="O", board_size=19)
        payload = {
            "m": [["e10", 0.9], ["f14", 0.8], ["g13", 0.7]]
        }
        local_moves_empty, tenuki_move_empty, _rank_by_move_empty = jd._select_candidates_from_root_payload(
            family="O",
            board_size=19,
            ply=0,
            entries=(),
            payload={"moves": []},
            red=set(red),
            blue=set(blue),
        )
        local_moves_full, tenuki_move_full, _rank_by_move_full = jd._select_candidates_from_root_payload(
            family="O",
            board_size=19,
            ply=0,
            entries=(),
            payload=payload,
            red=set(red),
            blue=set(blue),
        )
        self.assertEqual(local_moves_full, local_moves_empty)
        self.assertTrue(local_moves_full)
        self.assertIsNone(tenuki_move_empty)
        self.assertIsNone(tenuki_move_full)

    def test_selects_best_outside_local_tenuki_by_sequence_order(self):
        red, blue = jd._balance_cells(family="A", board_size=19)
        payload = {
            "m": [["p15", 0.20], ["f9", 0.40], ["e10", 0.50]]
        }
        local_moves, tenuki_move, meta_by_move = jd._select_candidates_from_root_payload(
            family="A",
            board_size=19,
            ply=1,
            entries=((5, 4),),
            payload=payload,
            red=set(red),
            blue=set(blue),
        )
        self.assertEqual(tenuki_move, "f9")
        self.assertEqual(local_moves[0][0], "p15")
        self.assertIsNone(meta_by_move["f9"]["cleaned_rank"])

    def test_acute_dead_region_moves_are_filtered_before_candidate_selection(self):
        payload = {
            "m": [["s19", 0.50], ["p17", 0.40]]
        }
        local_moves, tenuki_move, meta_by_move = jd._select_candidates_from_root_payload(
            family="A",
            board_size=19,
            ply=2,
            entries=((4, 2), (2, 3)),
            payload=payload,
            red={jd.parse_cell("r16")},
            blue={jd.parse_cell("q18")},
        )
        self.assertEqual(local_moves, [("p17", (3, 4))])
        self.assertIsNone(tenuki_move)
        self.assertEqual(meta_by_move["p17"]["cleaned_rank"], 1)

    def test_acute_equivalent_move_is_canonicalized_before_candidate_selection(self):
        payload = {
            "m": [["q18", 0.50], ["r17", 0.40], ["p16", 0.30]]
        }
        local_moves, tenuki_move, meta_by_move = jd._select_candidates_from_root_payload(
            family="A",
            board_size=19,
            ply=3,
            entries=((4, 2), (3, 3), (4, 3)),
            payload=payload,
            red={jd.parse_cell("r16"), jd.parse_cell("q16")},
            blue={jd.parse_cell("q17")},
        )
        self.assertEqual(local_moves, [("r17", (3, 2)), ("p16", (4, 4))])
        self.assertIsNone(tenuki_move)
        self.assertEqual(meta_by_move["r17"]["cleaned_rank"], 1)
        self.assertEqual(meta_by_move["p16"]["cleaned_rank"], 2)

    def test_line_importance_prunes_otherwise_strong_child(self):
        line_importance_min = 0.75
        importance = line_importance_min + 0.01
        node = jd.JosekiNode(family="A", entries=((5, 4),), realized_moves=("p15",), importance=importance)
        local_meta_by_cell = {
            "m1": {"local": (1, 1), "is_forced": False},
            "m2": {"local": (2, 1), "is_forced": False},
        }
        child_positions_by_move = {"j1": "pass", "m1": "m1", "m2": "m2"}
        almost_kept = (line_importance_min / importance) - 0.01
        analyze_payloads = {
            "pass": {"r": 0.00},
            "m1": {"r": 1.00},
            "m2": {"r": almost_kept},
        }
        with (
            mock.patch.object(jd, "_line_importance_min", return_value=line_importance_min),
            mock.patch.object(jd, "_ply_decay", return_value=0.99),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                (PARENT_POS, "red", None, local_meta_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual([child.line for child in children], ["A[5,4:1,1]"])
        self.assertEqual(record["retained_lines"], ["A[5,4:1,1]"])

    def test_root_local_children_use_configured_importance_overrides(self):
        node = jd.JosekiNode(family="O", entries=(), realized_moves=(), importance=1.0)
        local_meta_by_cell = {
            "m1": {"local": (4, 4), "is_forced": False},
            "m2": {"local": (5, 5), "is_forced": False},
        }
        child_positions_by_move = {"m1": "m1", "m2": "m2"}
        analyze_payloads = {
            "m1": {"r": 0.60},
            "m2": {"r": 0.20},
        }
        with (
            mock.patch.object(jd, "_line_importance_min", return_value=0.0),
            mock.patch.object(jd, "_ply_decay", return_value=0.98),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.85),
            mock.patch.object(
                jd,
                "JOSEKI_CHILD_OVERRIDE_RULES",
                [("O", (), (4, 4), 0.61), ("O", (), (5, 5), 0.73)],
            ),
            mock.patch.object(
                jd.lps,
                "_aggregate_moves",
                return_value=[
                    {"move": "m1", "stone_fraction": 1.0},
                    {"move": "m2", "stone_fraction": 0.5},
                ],
            ),
        ):
            _record, children = jd._finalize_node_expansion(
                node,
                (PARENT_POS, "red", None, local_meta_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual(
            [round(child.importance, 3) for child in children],
            [0.715, 0.598],
        )

    def test_child_cap_uses_child_importance_not_raw_stone_fraction(self):
        node = jd.JosekiNode(family="A", entries=((5, 4),), realized_moves=("p15",), importance=1.0)
        local_meta_by_cell = {"m1": {"local": (1, 1), "is_forced": False}}
        child_positions_by_move = {"j1": "tenuki", "m1": "local"}
        analyze_payloads = {
            "tenuki": {"r": 1.00},
            "local": {"r": 0.99},
        }
        with (
            mock.patch.object(jd, "_max_children_for_ply", return_value=1),
            mock.patch.object(jd, "_line_importance_min", return_value=0.0),
            mock.patch.object(jd, "_ply_decay", return_value=0.98),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.0),
            mock.patch.object(jd, "TENUKI_STONE_FRACTION_MIN", 0.0),
            mock.patch.object(jd, "TENUKI_IMPORTANCE_MULT", 0.5),
            mock.patch.object(
                jd.lps,
                "_aggregate_moves",
                return_value=[
                    {"move": "j1", "stone_fraction": 1.0},
                    {"move": "m1", "stone_fraction": 0.9},
                ],
            ),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                (PARENT_POS, "red", "j1", local_meta_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual([child.line for child in children], ["A[5,4:1,1]"])
        self.assertEqual(record["retained_lines"], ["A[5,4:1,1]"])

    def test_override_children_are_exempt_from_cap(self):
        node = jd.JosekiNode(family="O", entries=((4, 4),), realized_moves=("d16",), importance=1.0)
        with (
            mock.patch.object(jd, "_max_children_for_ply", return_value=1),
            mock.patch.object(jd, "_line_importance_min", return_value=0.0),
            mock.patch.object(jd, "_ply_decay", return_value=1.0),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.85),
            mock.patch.object(
                jd,
                "_select_candidates_from_root_payload",
                return_value=([("e17", (3, 3))], None, {}),
            ),
            mock.patch.object(
                jd,
                "JOSEKI_CHILD_OVERRIDE_RULES",
                [("O", ((4, 4),), (2, 4), 0.90)],
            ),
            mock.patch.object(
                jd.lps,
                "_aggregate_moves",
                return_value=[
                    {"move": "d18", "stone_fraction": 0.10},
                    {"move": "e17", "stone_fraction": 0.95},
                ],
            ),
        ):
            prepared = jd._prepare_node(
                node,
                board_size=19,
                root_payload={},
            )
            self.assertEqual(prepared[3]["d18"]["local"], (2, 4))
            self.assertEqual(prepared[3]["d18"]["is_forced"], True)
            self.assertEqual(prepared[3]["e17"]["local"], (3, 3))
            record, children = jd._finalize_node_expansion(
                node,
                prepared,
                analyze_payloads={child_position: {"r": 0.5} for child_position in prepared[4].values()},
            )
        self.assertEqual([child.line for child in children], ["O[4,4:3,3]", "O[4,4:2,4]"])
        self.assertEqual(record["retained_lines"], ["O[4,4:3,3]", "O[4,4:2,4]"])

    def test_second_consecutive_tenuki_is_scored_but_not_retained(self):
        node = jd.JosekiNode(family="A", entries=((5, 4), None), realized_moves=("p15", "j1"), importance=1.0)
        local_meta_by_cell = {"m1": {"local": (1, 1), "is_forced": False}}
        child_positions_by_move = {"j1": "pass", "m1": "local1", "t1": "tenuki"}
        analyze_payloads = {
            "pass": {"r": 0.50},
            "local1": {"r": 0.60},
            "tenuki": {"r": 0.80},
        }
        with (
            mock.patch.object(jd, "_line_importance_min", return_value=0.0),
            mock.patch.object(jd, "_ply_decay", return_value=1.0),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.0),
            mock.patch.object(jd, "TENUKI_STONE_FRACTION_MIN", 0.0),
            mock.patch.object(
                jd.lps,
                "_aggregate_moves",
                return_value=[
                    {"move": "j1", "stone_fraction": 0.0, "mean_winrate": 0.50},
                    {"move": "m1", "stone_fraction": 0.75, "mean_winrate": 0.60},
                    {"move": "t1", "stone_fraction": 1.0, "mean_winrate": 0.80},
                ],
            ),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                (PARENT_POS, "red", "t1", local_meta_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual([child.line for child in children], ["A[5,4::1,1]"])
        self.assertEqual(record["retained_lines"], ["A[5,4::1,1]"])
        self.assertIn({"kind": "tenuki", "stone_fraction": 1.0}, record["candidates"])

    def test_suppressed_second_tenuki_still_sets_tail_baseline(self):
        node = jd.JosekiNode(family="A", entries=((5, 4), None), realized_moves=("p15", "j1"), importance=1.0)
        local_meta_by_cell = {
            "m1": {"local": (1, 1), "cleaned_rank": 1, "prior": 0.20, "is_forced": False},
            "m2": {"local": (2, 1), "cleaned_rank": 2, "prior": 0.10, "is_forced": False},
        }
        child_positions_by_move = {"j1": "pass", "m1": "local1", "m2": "local2", "t1": "tenuki"}
        analyze_payloads = {
            "pass": {"r": 0.50},
            "local1": {"r": 0.60},
            "local2": {"r": 0.70},
            "tenuki": {"r": 0.80},
        }
        with (
            mock.patch.object(jd, "_max_children_for_ply", return_value=1),
            mock.patch.object(jd, "_line_importance_min", return_value=0.0),
            mock.patch.object(jd, "_ply_decay", return_value=1.0),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.0),
            mock.patch.object(jd, "TENUKI_STONE_FRACTION_MIN", 0.0),
            mock.patch.object(
                jd.lps,
                "_aggregate_moves",
                return_value=[
                    {"move": "j1", "stone_fraction": 0.0, "mean_winrate": 0.50},
                    {"move": "m1", "stone_fraction": 0.90, "mean_winrate": 0.60},
                    {"move": "m2", "stone_fraction": 0.80, "mean_winrate": 0.70},
                    {"move": "t1", "stone_fraction": 1.0, "mean_winrate": 0.80},
                ],
            ),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                (PARENT_POS, "red", "t1", local_meta_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual({child.line for child in children}, {"A[5,4::1,1]", "A[5,4::2,1]"})
        by_local = {tuple(candidate["local"]): candidate for candidate in record["candidates"] if candidate["kind"] == "local"}
        self.assertLess(by_local[(2, 1)]["stone_fraction"], 1.0)

    def test_outside_top_k_local_is_penalized_and_does_not_set_baseline(self):
        node = jd.JosekiNode(family="A", entries=((5, 4),), realized_moves=("p15",), importance=1.0)
        local_meta_by_cell = {
            "m1": {"local": (1, 1), "cleaned_rank": 1, "prior": 0.20, "is_forced": False},
            "m2": {"local": (2, 1), "cleaned_rank": 2, "prior": 0.10, "is_forced": False},
        }
        child_positions_by_move = {"j1": "pass", "m1": "local1", "m2": "local2"}
        analyze_payloads = {
            "pass": {"r": 0.50},
            "local1": {"r": 0.60},
            "local2": {"r": 0.80},
        }
        with (
            mock.patch.object(jd, "_max_children_for_ply", return_value=1),
            mock.patch.object(jd, "_ply_decay", return_value=1.0),
            mock.patch.object(jd, "_line_importance_min", return_value=0.5),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.0),
            mock.patch.object(
                jd.lps,
                "_aggregate_moves",
                return_value=[
                    {"move": "j1", "stone_fraction": 0.0, "mean_winrate": 0.50},
                    {"move": "m1", "stone_fraction": 0.90, "mean_winrate": 0.60},
                    {"move": "m2", "stone_fraction": 0.80, "mean_winrate": 0.80},
                ],
            ),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                (PARENT_POS, "red", None, local_meta_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
            penalized_weight = jd._outside_top_k_candidate_weight(
                node,
                meta=local_meta_by_cell["m2"],
                retained_local_count=1,
            )
        self.assertEqual({child.line for child in children}, {"A[5,4:1,1]", "A[5,4:2,1]"})
        by_local = {tuple(candidate["local"]): candidate for candidate in record["candidates"] if candidate["kind"] == "local"}
        self.assertEqual(by_local[(1, 1)]["stone_fraction"], 0.9)
        self.assertEqual(by_local[(2, 1)]["stone_fraction"], 1.0)
        self.assertLess(penalized_weight, 1.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from unittest import mock

import joseki_database as jd


class JosekiDatabaseTests(unittest.TestCase):
    def test_cached_multi_position_analyze_reuses_cached_payloads(self):
        pos1 = "https://hexworld.org/board/#19c1,a1d2"
        pos2 = "https://hexworld.org/board/#19c1,a1d2p15"
        cache = {jd.lps._cache_key(pos1): {"root_eval": {"red_winrate": 0.5}, "moves": [{"move": "p15", "rank": 1, "prior": 0.2}]}}
        with mock.patch(
            "joseki_database.lps._run_multi_position_analyze",
            return_value={pos2: {"ok": True, "input": pos2, "root_eval": {"red_winrate": 0.6}, "moves": [{"move": "q14", "rank": 2, "prior": 0.1}], "meta": {"elapsed_ms": 1}}},
        ) as mocked:
            payloads = jd._run_multi_position_analyze_cached(
                hexata_main=Path("dummy"),
                position_inputs=[pos1, pos2, pos2],
                raw_nn_cache=cache,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(payloads[pos1]["root_eval"]["red_winrate"], 0.5)
        self.assertEqual(payloads[pos2]["root_eval"]["red_winrate"], 0.6)

    def test_root_ply_uses_hardcoded_local_moves_and_no_tenuki(self):
        red, blue = jd._balance_cells(family="O", board_size=19)
        payload = {
            "moves": [
                {"move": "e10", "rank": 1, "prior": 0.9},
                {"move": "f14", "rank": 2, "prior": 0.8},
                {"move": "g13", "rank": 3, "prior": 0.7},
            ]
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

    def test_selects_best_outside_local_tenuki_by_rank(self):
        red, blue = jd._balance_cells(family="A", board_size=19)
        payload = {
            "moves": [
                {"move": "e10", "rank": 5, "prior": 0.50},
                {"move": "p15", "rank": 1, "prior": 0.20},
                {"move": "f9", "rank": 2, "prior": 0.40},
            ]
        }
        local_moves, tenuki_move, rank_by_move = jd._select_candidates_from_root_payload(
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
        self.assertEqual(rank_by_move["f9"], 2)

    def test_acute_dead_region_moves_are_filtered_before_candidate_selection(self):
        payload = {
            "moves": [
                {"move": "s19", "rank": 1, "prior": 0.50},
                {"move": "p17", "rank": 2, "prior": 0.40},
            ]
        }
        local_moves, tenuki_move, rank_by_move = jd._select_candidates_from_root_payload(
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
        self.assertEqual(rank_by_move, {"p17": 2})

    def test_acute_equivalent_move_is_canonicalized_before_candidate_selection(self):
        payload = {
            "moves": [
                {"move": "q18", "rank": 1, "prior": 0.50},
                {"move": "r17", "rank": 2, "prior": 0.40},
                {"move": "p16", "rank": 3, "prior": 0.30},
            ]
        }
        local_moves, tenuki_move, rank_by_move = jd._select_candidates_from_root_payload(
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
        self.assertEqual(rank_by_move, {"r17": 1, "p16": 3})

    def test_line_importance_prunes_otherwise_strong_child(self):
        line_importance_min = 0.75
        importance = line_importance_min + 0.01
        node = jd.JosekiNode(family="A", entries=((5, 4),), realized_moves=("p15",), importance=importance)
        local_by_cell = {"m1": (1, 1), "m2": (2, 1)}
        child_positions_by_move = {"j1": "pass", "m1": "m1", "m2": "m2"}
        almost_kept = (line_importance_min / importance) - 0.01
        analyze_payloads = {
            "pass": {"root_eval": {"red_winrate": 0.00}},
            "m1": {"root_eval": {"red_winrate": 1.00}},
            "m2": {"root_eval": {"red_winrate": almost_kept}},
        }
        with (
            mock.patch.object(jd, "_line_importance_min", return_value=line_importance_min),
            mock.patch.object(jd, "_ply_decay", return_value=0.99),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                ("parent", "red", None, local_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual([child.line for child in children], ["A[5,4:1,1]"])
        self.assertEqual(record["retained_lines"], ["A[5,4:1,1]"])

    def test_root_local_children_use_configured_importance_overrides(self):
        node = jd.JosekiNode(family="O", entries=(), realized_moves=(), importance=1.0)
        local_by_cell = {"m1": (4, 4), "m2": (5, 5)}
        child_positions_by_move = {"m1": "m1", "m2": "m2"}
        analyze_payloads = {
            "m1": {"root_eval": {"red_winrate": 0.60}},
            "m2": {"root_eval": {"red_winrate": 0.20}},
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
                ("parent", "red", None, local_by_cell, child_positions_by_move),
                analyze_payloads=analyze_payloads,
            )
        self.assertEqual(
            [round(child.importance, 3) for child in children],
            [0.715, 0.598],
        )

    def test_child_cap_uses_child_importance_not_raw_stone_fraction(self):
        node = jd.JosekiNode(family="A", entries=((5, 4),), realized_moves=("p15",), importance=1.0)
        local_by_cell = {"m1": (1, 1)}
        child_positions_by_move = {"j1": "tenuki", "m1": "local"}
        analyze_payloads = {
            "tenuki": {"root_eval": {"red_winrate": 1.00}},
            "local": {"root_eval": {"red_winrate": 0.99}},
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
                ("parent", "red", "j1", local_by_cell, child_positions_by_move),
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
            self.assertEqual(prepared[3]["d18"], (2, 4))
            self.assertEqual(prepared[3]["e17"], (3, 3))
            record, children = jd._finalize_node_expansion(
                node,
                prepared,
                analyze_payloads={child_position: {"root_eval": {"red_winrate": 0.5}} for child_position in prepared[4].values()},
            )
        self.assertEqual([child.line for child in children], ["O[4,4:3,3]", "O[4,4:2,4]"])
        self.assertEqual(record["retained_lines"], ["O[4,4:3,3]", "O[4,4:2,4]"])

    def test_prune_prepared_to_non_local_children_keeps_pass_proxy_and_tenuki(self):
        prepared = (
            "parent",
            "red",
            "t1",
            {"m1": (1, 1), "m2": (2, 1)},
            {"j1": "pass", "m1": "local1", "m2": "local2", "t1": "tenuki"},
        )
        pruned = jd._prune_prepared_to_non_local_children(prepared)
        self.assertEqual(pruned[2], "t1")
        self.assertEqual(pruned[3], {})
        self.assertEqual(pruned[4], {"j1": "pass", "t1": "tenuki"})


if __name__ == "__main__":
    unittest.main()

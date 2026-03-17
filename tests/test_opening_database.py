import unittest
from unittest import mock

import opening_database as od


def _winrate_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + (10.0 ** (-float(elo) / 400.0)))


class OpeningDatabaseTests(unittest.TestCase):
    def test_canonical_fair_root_move_collapses_rotation_and_row2_bucket(self):
        self.assertEqual(od._canonical_fair_root_move("f9", board_size=11), "f3")
        self.assertEqual(od._canonical_fair_root_move("e9", board_size=11), "g3")
        self.assertEqual(od._canonical_fair_root_move("d9", board_size=11), "h3")
        self.assertEqual(od._canonical_fair_root_move("c10", board_size=11), "i2")
        self.assertEqual(od._canonical_fair_root_move("k5", board_size=11), "a7")
        self.assertEqual(od._canonical_fair_root_move("k1", board_size=11), "a11")
        self.assertEqual(od._canonical_fair_root_move("a2", board_size=11), "c2")
        self.assertEqual(od._canonical_fair_root_move("b2", board_size=11), "c2")

    def test_derive_fair_root_study_uses_reference_calibration(self):
        ref_elo = 400.0
        payload = {
            "moves": [
                {"move": "b10", "red_winrate": _winrate_from_elo(ref_elo)},
                {"move": "c2", "red_winrate": _winrate_from_elo(-160.0)},
                {"move": "f3", "red_winrate": _winrate_from_elo(0.0)},
                {"move": "a7", "red_winrate": _winrate_from_elo(240.0)},
            ]
        }
        with mock.patch.object(od, "_canonical_fair_root_representatives", return_value=("c2", "f3", "a7")):
            study = od._derive_fair_root_study(board_size=11, sweep_payload=payload)
        self.assertEqual(study["reference_move"], "b10")
        self.assertEqual(study["root_openings"], ["c2", "f3"])

    def test_select_root_candidates_uses_supplied_openings(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,",
            moves=(),
        )
        candidates = od._select_root_candidates(node=node, board_size=11, root_openings=("c2", "f3"))
        self.assertEqual([row["move"] for row in candidates], ["c2", "f3"])

    def test_select_top_prior_candidates_excludes_dead_region_moves(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,b4c2",
            moves=("b4", "c2"),
        )
        payload = {
            "moves": [
                {"move": "a1", "rank": 1, "prior": 0.25},
                {"move": "a2", "rank": 2, "prior": 0.22},
                {"move": "b3", "rank": 3, "prior": 0.2},
                {"move": "c3", "rank": 4, "prior": 0.18},
                {"move": "d4", "rank": 5, "prior": 0.16},
                {"move": "e5", "rank": 6, "prior": 0.14},
                {"move": "f6", "rank": 7, "prior": 0.12},
                {"move": "g7", "rank": 8, "prior": 0.1},
            ]
        }
        candidates = od._select_top_prior_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["c3", "d4", "e5", "f6", "g7"])

    def test_select_top_prior_candidates_canonicalizes_equivalent_move_to_b3(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,b4c3c4",
            moves=("b4", "c3", "c4"),
        )
        payload = {
            "moves": [
                {"move": "c2", "rank": 1, "prior": 0.25},
                {"move": "b3", "rank": 2, "prior": 0.22},
                {"move": "d4", "rank": 3, "prior": 0.2},
            ]
        }
        candidates = od._select_top_prior_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["b3", "d4"])

    def test_select_top_prior_candidates_keeps_high_prior_beyond_top_k(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,a1b2c3d4e5",
            moves=("a1", "b2", "c3", "d4", "e5"),
        )
        payload = {
            "moves": [
                {"move": "f6", "rank": 1, "prior": 0.30},
                {"move": "g7", "rank": 2, "prior": 0.25},
                {"move": "h8", "rank": 3, "prior": 0.21},
                {"move": "i9", "rank": 4, "prior": 0.19},
            ]
        }
        with mock.patch.object(od, "_extra_candidate_prior_min", return_value=0.20):
            candidates = od._select_top_prior_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["f6", "g7", "h8"])

    def test_mover_winrate_inverts_for_blue_parent(self):
        payload = {"root_eval": {"red_winrate": 0.7}}
        self.assertAlmostEqual(
            od._mover_winrate_from_child_payload(child_payload=payload, parent_to_play="red"),
            0.7,
        )
        self.assertAlmostEqual(
            od._mover_winrate_from_child_payload(child_payload=payload, parent_to_play="blue"),
            0.3,
        )

    def test_root_stone_fraction_uses_sqrt_of_best_fixed_root_distance(self):
        root_study = {
            "rows": [
                {"move": "g3", "stone_fraction": 0.53},
                {"move": "h4", "stone_fraction": 0.42},
                {"move": "j5", "stone_fraction": 0.61},
            ]
        }
        self.assertAlmostEqual(
            od._root_stone_fraction_from_study(move="g3", root_study=root_study),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            od._root_stone_fraction_from_study(move="h4", root_study=root_study),
            0.95 ** 0.5,
            places=6,
        )
        self.assertAlmostEqual(
            od._root_stone_fraction_from_study(move="j5", root_study=root_study),
            0.92 ** 0.5,
            places=6,
        )

    def test_finalize_node_retains_by_importance_threshold(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,a3",
            moves=("a3",),
            importance=0.98,
        )
        best_wr = 0.75
        best_elo = od._winrate_to_elo(best_wr)
        full_stone_elo = 1000.0
        near_wr = _winrate_from_elo(best_elo - 100.0)
        far_wr = _winrate_from_elo(best_elo - 120.0)
        candidates = [
            {
                "move": "b4",
                "rank": 1,
                "prior": 0.4,
                "child_position": "p1",
                "parent_to_play": "blue",
                "board_size": 11,
            },
            {
                "move": "c4",
                "rank": 2,
                "prior": 0.1,
                "child_position": "p2",
                "parent_to_play": "blue",
                "board_size": 11,
            },
            {
                "move": "d4",
                "rank": 3,
                "prior": 0.05,
                "child_position": "p3",
                "parent_to_play": "blue",
                "board_size": 11,
            },
        ]
        child_payloads = {
            "p1": {"root_eval": {"red_winrate": 1.0 - best_wr}},
            "p2": {"root_eval": {"red_winrate": 1.0 - near_wr}},
            "p3": {"root_eval": {"red_winrate": 1.0 - far_wr}},
        }
        root_study = {
            "reference_elo": full_stone_elo / 4.0,
            "rows": [{"move": "a1", "stone_fraction": 0.53}],
        }
        with mock.patch.object(od, "_importance_min", return_value=0.86), mock.patch.object(od, "_ply_decay", return_value=0.98):
            record, children = od._finalize_node(
                node,
                candidates=candidates,
                child_payloads=child_payloads,
                root_study=root_study,
            )
        by_move = {row["move"]: row for row in record["candidates"]}
        self.assertAlmostEqual(record["importance"], 0.98, places=6)
        self.assertTrue(by_move["b4"]["retained"])
        self.assertTrue(by_move["c4"]["retained"])
        self.assertFalse(by_move["d4"]["retained"])
        self.assertAlmostEqual(by_move["c4"]["elo_loss"], 100.0, delta=0.01)
        self.assertAlmostEqual(by_move["d4"]["elo_loss"], 120.0, delta=0.01)
        self.assertAlmostEqual(by_move["c4"]["stone_fraction"], 0.9, places=6)
        self.assertAlmostEqual(by_move["c4"]["importance"], 0.86436, places=6)
        self.assertAlmostEqual(by_move["d4"]["stone_fraction"], 0.88, places=6)
        self.assertAlmostEqual(by_move["d4"]["importance"], 0.845152, places=6)
        self.assertEqual([child.moves for child in children], [("a3", "b4"), ("a3", "c4")])

    def test_finalize_node_uses_root_stone_fraction_for_curated_openings(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,",
            moves=(),
        )
        candidates = od._select_root_candidates(node=node, board_size=11, root_openings=("c2", "f3", "a7"))
        best_wr = 0.74
        lower_wr = _winrate_from_elo(od._winrate_to_elo(best_wr) - 120.0)
        payloads = {
            candidates[0]["child_position"]: {"root_eval": {"red_winrate": best_wr}},
            candidates[1]["child_position"]: {"root_eval": {"red_winrate": lower_wr}},
        }
        for cand in candidates[2:]:
            payloads[cand["child_position"]] = {"root_eval": {"red_winrate": 0.5}}
        root_study = {
            "reference_elo": 400.0,
            "rows": [
                {"move": "c2", "stone_fraction": 0.53},
                {"move": "f3", "stone_fraction": 0.42},
                {"move": "a7", "stone_fraction": 0.61},
            ],
        }
        with mock.patch.object(od, "_importance_min", return_value=0.95), mock.patch.object(od, "_ply_decay", return_value=0.98):
            record, children = od._finalize_node(
                node,
                candidates=candidates,
                child_payloads=payloads,
                root_study=root_study,
            )
        by_move = {row["move"]: row for row in record["candidates"]}
        self.assertAlmostEqual(record["importance"], 1.0, places=6)
        self.assertAlmostEqual(by_move[candidates[0]["move"]]["mover_winrate"], best_wr, places=6)
        self.assertAlmostEqual(by_move[candidates[1]["move"]]["elo_loss"], 120.0, delta=0.01)
        self.assertAlmostEqual(by_move["c2"]["stone_fraction"], 1.0, places=6)
        self.assertAlmostEqual(by_move["f3"]["stone_fraction"], 0.95 ** 0.5, places=6)
        self.assertAlmostEqual(by_move["a7"]["stone_fraction"], 0.92 ** 0.5, places=6)
        self.assertEqual([child.moves for child in children], [("c2",), ("f3",)])

    def test_can_skip_child_expansion_uses_upper_bound(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,a3",
            moves=("a3",),
            importance=0.9,
        )
        with (
            mock.patch.object(od, "_importance_min", return_value=0.86),
            mock.patch.object(od, "_ply_decay", return_value=0.95),
        ):
            self.assertTrue(od._can_skip_child_expansion(node=node, board_size=11))
        with (
            mock.patch.object(od, "_importance_min", return_value=0.85),
            mock.patch.object(od, "_ply_decay", return_value=0.95),
        ):
            self.assertFalse(od._can_skip_child_expansion(node=node, board_size=11))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

import opening_database as od


def _winrate_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + (10.0 ** (-float(elo) / 400.0)))


class OpeningDatabaseTests(unittest.TestCase):
    def test_build_output_payload_includes_mode(self):
        payload = od._build_output_payload(
            board_size=11,
            root_openings=("c2", "f3"),
            root_study={"reference_elo": 100.0},
            nodes=[],
            completed=False,
            completed_ply=2,
        )
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["mode"], "openings")

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
            "m": [
                ["b10", _winrate_from_elo(ref_elo)],
                ["c2", _winrate_from_elo(-160.0)],
                ["f3", _winrate_from_elo(0.0)],
                ["a7", _winrate_from_elo(240.0)],
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

    def test_select_child_evaluation_candidates_excludes_dead_region_moves(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,b4c2",
            moves=("b4", "c2"),
        )
        payload = {
            "m": [
                ["a1", 0.25],
                ["a2", 0.22],
                ["b3", 0.2],
                ["c3", 0.18],
                ["d4", 0.16],
                ["e5", 0.14],
                ["f6", 0.12],
                ["g7", 0.1],
            ]
        }
        with mock.patch.object(od, "_importance_min", return_value=0.0):
            candidates = od._select_child_evaluation_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["c3", "d4", "e5", "f6", "g7"])

    def test_select_child_evaluation_candidates_canonicalizes_equivalent_move_to_b3(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,b4c3c4",
            moves=("b4", "c3", "c4"),
        )
        payload = {
            "m": [["c2", 0.25], ["b3", 0.22], ["d4", 0.2]]
        }
        with mock.patch.object(od, "_importance_min", return_value=0.0):
            candidates = od._select_child_evaluation_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["b3", "d4"])

    def test_select_child_evaluation_candidates_keeps_high_prior_beyond_top_k(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,a1b2c3d4e5",
            moves=("a1", "b2", "c3", "d4", "e5"),
        )
        payload = {
            "m": [["f6", 0.30], ["g7", 0.25], ["h8", 0.21], ["i9", 0.19]]
        }
        with (
            mock.patch.object(od, "_extra_candidate_prior_min", return_value=0.20),
            mock.patch.object(od, "_importance_min", return_value=0.0),
        ):
            candidates = od._select_child_evaluation_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["f6", "g7", "h8", "i9"])

    def test_select_child_evaluation_candidates_skips_outside_top_k_when_penalized_upper_bound_cannot_retain(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,a1b2c3d4e5",
            moves=("a1", "b2", "c3", "d4", "e5"),
            importance=0.89,
        )
        payload = {
            "m": [["f6", 0.30], ["g7", 0.25], ["h8", 0.19], ["i9", 0.21]]
        }
        test_floor = 0.88
        with (
            mock.patch.object(od, "_top_k_for_ply", return_value=2),
            mock.patch.object(od, "_extra_candidate_prior_min", return_value=0.20),
            mock.patch.object(od, "_ply_decay", return_value=1.0),
            mock.patch.object(od, "_importance_min", return_value=test_floor),
            mock.patch.object(od, "_outside_top_k_prior_log_step", return_value=0.23),
            mock.patch.object(od, "_outside_top_k_exponent_rank_step", return_value=0.04),
            mock.patch.object(od, "_outside_top_k_exponent_ply_step", return_value=0.02),
        ):
            candidates = od._select_child_evaluation_candidates(node=node, payload=payload)
        self.assertEqual([row["move"] for row in candidates], ["f6", "g7", "i9"])

    def test_mover_winrate_inverts_for_blue_parent(self):
        payload = {"r": 0.7}
        self.assertAlmostEqual(
            od._mover_winrate_from_child_payload(child_payload=payload, parent_to_play="red"),
            0.7,
        )
        self.assertAlmostEqual(
            od._mover_winrate_from_child_payload(child_payload=payload, parent_to_play="blue"),
            0.3,
        )

    def test_root_stone_fraction_uses_sqrt_of_absolute_distance_from_fair(self):
        root_study = {
            "rows": [
                {"move": "g3", "stone_fraction": 0.53},
                {"move": "h4", "stone_fraction": 0.42},
                {"move": "j5", "stone_fraction": 0.61},
            ]
        }
        self.assertAlmostEqual(
            od._root_stone_fraction_from_study(move="g3", root_study=root_study),
            0.97 ** 0.5,
            places=6,
        )
        self.assertAlmostEqual(
            od._root_stone_fraction_from_study(move="h4", root_study=root_study),
            0.92 ** 0.5,
            places=6,
        )
        self.assertAlmostEqual(
            od._root_stone_fraction_from_study(move="j5", root_study=root_study),
            0.89 ** 0.5,
            places=6,
        )

    def test_finalize_node_uses_only_anchor_candidates_for_best_elo_baseline(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,a3",
            moves=("a3",),
            importance=0.98,
        )
        anchor_best_wr = 0.75
        anchor_second_wr = _winrate_from_elo(od._winrate_to_elo(anchor_best_wr) - 100.0)
        outside_top_k_best_wr = _winrate_from_elo(od._winrate_to_elo(anchor_best_wr) + 60.0)
        candidates = [
            {
                "move": "b4",
                "rank": 1,
                "cleaned_rank": 1,
                "prior": 0.4,
                "child_position": "p1",
                "parent_to_play": "red",
                "board_size": 11,
            },
            {
                "move": "c4",
                "rank": 2,
                "cleaned_rank": 2,
                "prior": 0.1,
                "child_position": "p2",
                "parent_to_play": "red",
                "board_size": 11,
            },
            {
                "move": "d4",
                "rank": 3,
                "cleaned_rank": 3,
                "prior": 0.05,
                "child_position": "p3",
                "parent_to_play": "red",
                "board_size": 11,
            },
        ]
        child_payloads = {
            "p1": {"r": anchor_best_wr},
            "p2": {"r": anchor_second_wr},
            "p3": {"r": outside_top_k_best_wr},
        }
        root_study = {
            "reference_elo": 250.0,
            "rows": [{"move": "a1", "stone_fraction": 0.53}],
        }
        test_floor = 0.845
        test_prior_log_step = 0.27
        test_rank_step = 0.04
        test_ply_step = 0.03
        with (
            mock.patch.object(od, "_importance_min", return_value=test_floor),
            mock.patch.object(od, "_ply_decay", return_value=0.98),
            mock.patch.object(od, "_top_k_for_ply", return_value=2),
            mock.patch.object(od, "_outside_top_k_prior_log_step", return_value=test_prior_log_step),
            mock.patch.object(od, "_outside_top_k_exponent_rank_step", return_value=test_rank_step),
            mock.patch.object(od, "_outside_top_k_exponent_ply_step", return_value=test_ply_step),
        ):
            record, children = od._finalize_node(
                node,
                candidates=candidates,
                child_payloads=child_payloads,
                root_study=root_study,
            )
        by_move = {row["move"]: row for row in record["candidates"]}
        expected_penalty = test_floor ** (test_prior_log_step * (-od.math.log10(0.05)))
        self.assertAlmostEqual(by_move["b4"]["elo_loss"], 0.0, places=6)
        self.assertAlmostEqual(by_move["c4"]["elo_loss"], 100.0, delta=0.01)
        self.assertAlmostEqual(by_move["d4"]["elo_loss"], 0.0, places=6)
        self.assertAlmostEqual(by_move["b4"]["candidate_weight"], 1.0, places=6)
        self.assertAlmostEqual(by_move["d4"]["candidate_weight"], expected_penalty, places=6)
        self.assertAlmostEqual(by_move["c4"]["stone_fraction"], 0.9, places=6)
        self.assertAlmostEqual(by_move["d4"]["stone_fraction"], 1.0, places=6)
        self.assertEqual([child.moves for child in children], [("a3", "b4"), ("a3", "c4"), ("a3", "d4")])

    def test_finalize_node_uses_root_stone_fraction_for_curated_openings(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,",
            moves=(),
        )
        candidates = od._select_root_candidates(node=node, board_size=11, root_openings=("c2", "f3", "a7"))
        best_wr = 0.74
        lower_wr = _winrate_from_elo(od._winrate_to_elo(best_wr) - 120.0)
        payloads = {
            candidates[0]["child_position"]: {"r": best_wr},
            candidates[1]["child_position"]: {"r": lower_wr},
        }
        for cand in candidates[2:]:
            payloads[cand["child_position"]] = {"r": 0.5}
        root_study = {
            "reference_elo": 400.0,
            "rows": [
                {"move": "c2", "stone_fraction": 0.53},
                {"move": "f3", "stone_fraction": 0.42},
                {"move": "a7", "stone_fraction": 0.61},
            ],
        }
        with (
            mock.patch.object(od, "_importance_min", return_value=0.95),
            mock.patch.object(od, "_ply_decay", return_value=0.98),
            mock.patch.object(od, "OPENING_ROOT_IMPORTANCE_OVERRIDES", []),
        ):
            record, children = od._finalize_node(
                node,
                candidates=candidates,
                child_payloads=payloads,
                root_study=root_study,
            )
        by_move = {row["move"]: row for row in record["candidates"]}
        self.assertAlmostEqual(record["importance"], 1.0, places=6)
        self.assertAlmostEqual(by_move[candidates[0]["move"]]["raw_mover_winrate"], best_wr, places=6)
        self.assertAlmostEqual(by_move[candidates[1]["move"]]["elo_loss"], 120.0, delta=0.01)
        self.assertAlmostEqual(by_move["c2"]["stone_fraction"], 0.97 ** 0.5, places=6)
        self.assertAlmostEqual(by_move["f3"]["stone_fraction"], 0.92 ** 0.5, places=6)
        self.assertAlmostEqual(by_move["a7"]["stone_fraction"], 0.89 ** 0.5, places=6)
        self.assertEqual([child.moves for child in children], [("c2",)])

    def test_finalize_node_uses_configured_root_importance_overrides(self):
        node = od.OpeningNode(
            position="https://hexworld.org/board/#11c1,",
            moves=(),
        )
        candidates = od._select_root_candidates(node=node, board_size=11, root_openings=("a6", "c2"))
        payloads = {
            cand["child_position"]: {"r": 0.6}
            for cand in candidates
        }
        root_study = {
            "reference_elo": 400.0,
            "rows": [
                {"move": "a6", "stone_fraction": 0.57},
                {"move": "c2", "stone_fraction": 0.10},
            ],
        }
        with (
            mock.patch.object(od, "_importance_min", return_value=0.92),
            mock.patch.object(od, "_ply_decay", return_value=0.98),
            mock.patch.object(od, "OPENING_ROOT_IMPORTANCE_OVERRIDES", [(11, "a6", 0.94)]),
        ):
            record, children = od._finalize_node(
                node,
                candidates=candidates,
                child_payloads=payloads,
                root_study=root_study,
            )
        by_move = {row["move"]: row for row in record["candidates"]}
        self.assertAlmostEqual(by_move["a6"]["stone_fraction"], 0.94, places=6)
        self.assertAlmostEqual(by_move["a6"]["importance"], 0.94 * 0.98, places=6)
        self.assertAlmostEqual(by_move["c2"]["stone_fraction"], 0.60 ** 0.5, places=6)
        self.assertEqual([child.moves for child in children], [("a6",)])

    def test_apply_prior_weighted_tree_values_propagates_retained_children(self):
        nodes = [
            {
                "parent": None,
                "move": None,
                "ply": 0,
                "importance": 1.0,
                "candidates": [
                    {
                        "move": "a3",
                        "child": 1,
                        "retained": True,
                        "prior": None,
                        "raw_mover_winrate": 0.74,
                    },
                    {
                        "move": "b4",
                        "child": None,
                        "retained": False,
                        "prior": None,
                        "raw_mover_winrate": 0.61,
                    },
                ],
            },
            {
                "parent": 0,
                "move": "a3",
                "ply": 1,
                "importance": 0.9,
                "candidates": [
                    {
                        "move": "d8",
                        "child": None,
                        "retained": True,
                        "prior": 0.75,
                        "raw_mover_winrate": 0.2,
                    },
                    {
                        "move": "f7",
                        "child": None,
                        "retained": True,
                        "prior": 0.25,
                        "raw_mover_winrate": 0.6,
                    },
                    {
                        "move": "g8",
                        "child": None,
                        "retained": False,
                        "prior": 0.1,
                        "raw_mover_winrate": 0.3,
                    },
                ],
            },
        ]
        od._apply_prior_weighted_tree_values(nodes=nodes)
        root_rows = {row["move"]: row for row in nodes[0]["candidates"]}
        child_rows = {row["move"]: row for row in nodes[1]["candidates"]}
        self.assertIsNone(nodes[0]["tree_red_winrate"])
        self.assertAlmostEqual(nodes[1]["tree_red_winrate"], 0.7, places=6)
        self.assertAlmostEqual(root_rows["a3"]["tree_mover_winrate"], 0.7, places=6)
        self.assertAlmostEqual(root_rows["b4"]["tree_mover_winrate"], 0.61, places=6)
        self.assertAlmostEqual(child_rows["d8"]["tree_mover_winrate"], 0.2, places=6)
        self.assertAlmostEqual(child_rows["f7"]["tree_mover_winrate"], 0.6, places=6)
        self.assertAlmostEqual(child_rows["g8"]["tree_mover_winrate"], 0.3, places=6)

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

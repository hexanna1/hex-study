import unittest
from unittest import mock

import joseki_database as jd

PARENT_POS = "https://hexworld.org/board/#19c1"


class JosekiDatabaseTests(unittest.TestCase):
    def test_suppressed_second_tenuki_still_sets_tail_baseline(self):
        node = jd.JosekiNode(family="A", entries=((5, 4), None), position=PARENT_POS, importance=1.0)
        local_meta_by_cell = {
            "m1": {"local": (1, 1), "cleaned_rank": 1, "prior": 0.20, "is_forced": False},
            "m2": {"local": (2, 1), "cleaned_rank": 2, "prior": 0.10, "is_forced": False},
        }
        child_positions_by_move = {"j1": "pass", "m1": "local1", "m2": "local2", "t1": "tenuki"}
        raw_nn_payloads = {
            "pass": {"r": 0.50},
            "local1": {"r": 0.60},
            "local2": {"r": 0.70},
            "tenuki": {"r": 0.80},
        }
        with (
            mock.patch.object(jd, "_top_k_for_ply", return_value=1),
            mock.patch.object(jd, "_line_importance_min", return_value=0.0),
            mock.patch.object(jd, "PLY_DECAY", 1.0),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.0),
            mock.patch.object(jd, "TENUKI_STONE_FRACTION_MIN", 0.0),
            mock.patch.object(
                jd,
                "_child_evaluation_rows",
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
                jd.NodeExpansion(PARENT_POS, "t1", local_meta_by_cell, child_positions_by_move),
                raw_nn_payloads=raw_nn_payloads,
            )
        self.assertEqual({child.line for child in children}, {"A[5,4::1,1]", "A[5,4::2,1]"})
        by_local = {tuple(candidate["local"]): candidate for candidate in record["candidates"] if candidate["kind"] == "local"}
        self.assertLess(by_local[(2, 1)]["stone_fraction"], 1.0)

    def test_outside_top_k_local_is_penalized_and_does_not_set_baseline(self):
        node = jd.JosekiNode(family="A", entries=((5, 4),), position=PARENT_POS, importance=1.0)
        local_meta_by_cell = {
            "m1": {"local": (1, 1), "cleaned_rank": 1, "prior": 0.20, "is_forced": False},
            "m2": {"local": (2, 1), "cleaned_rank": 2, "prior": 0.10, "is_forced": False},
        }
        child_positions_by_move = {"j1": "pass", "m1": "local1", "m2": "local2"}
        raw_nn_payloads = {
            "pass": {"r": 0.50},
            "local1": {"r": 0.60},
            "local2": {"r": 0.80},
        }
        with (
            mock.patch.object(jd, "_top_k_for_ply", return_value=1),
            mock.patch.object(jd, "PLY_DECAY", 1.0),
            mock.patch.object(jd, "_line_importance_min", return_value=0.5),
            mock.patch.object(jd, "STONE_FRACTION_MIN", 0.0),
            mock.patch.object(
                jd,
                "_child_evaluation_rows",
                return_value=[
                    {"move": "j1", "stone_fraction": 0.0, "mean_winrate": 0.50},
                    {"move": "m1", "stone_fraction": 0.90, "mean_winrate": 0.60},
                    {"move": "m2", "stone_fraction": 0.80, "mean_winrate": 0.80},
                ],
            ),
        ):
            record, children = jd._finalize_node_expansion(
                node,
                jd.NodeExpansion(PARENT_POS, None, local_meta_by_cell, child_positions_by_move),
                raw_nn_payloads=raw_nn_payloads,
            )
            penalized_weight = jd._outside_top_k_candidate_weight(
                node,
                meta=local_meta_by_cell["m2"],
                head_local_count=1,
            )
        self.assertEqual({child.line for child in children}, {"A[5,4:1,1]", "A[5,4:2,1]"})
        by_local = {tuple(candidate["local"]): candidate for candidate in record["candidates"] if candidate["kind"] == "local"}
        self.assertEqual(by_local[(1, 1)]["stone_fraction"], 0.9)
        self.assertEqual(by_local[(2, 1)]["stone_fraction"], 1.0)
        self.assertLess(penalized_weight, 1.0)

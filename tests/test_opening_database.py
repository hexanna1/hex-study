from pathlib import Path
import tempfile
import unittest
from unittest import mock

import opening_database as od


def _winrate_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + (10.0 ** (-float(elo) / 400.0)))


class OpeningDatabaseTests(unittest.TestCase):
    def test_merge_opening_child_uses_max_importance(self):
        children = {}
        first = od.OpeningNode(position="same", ply=3, importance=0.2, move="a1")
        stronger = od.OpeningNode(position="same", ply=3, importance=0.4, move="b1")

        od._merge_opening_child(children, child=first, parent=5)
        od._merge_opening_child(children, child=stronger, parent=8)

        self.assertEqual(list(children), ["same"])
        self.assertIs(children["same"], first)
        self.assertEqual(first.importance, 0.4)
        self.assertEqual((first.parent, first.move), (5, "a1"))

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

    def test_fair_root_sweep_uses_child_position_raw_nn_cache(self):
        root_position = od._empty_position(board_size=5)
        requested_moves = ("c2", "b4")
        child_payloads = {
            od.lps._position_after_move(root_position, "c2"): {"r": 0.25},
            od.lps._position_after_move(root_position, "b4"): {"r": 0.75},
        }
        raw_cache = {}
        with (
            mock.patch.object(od, "_canonical_fair_root_representatives", return_value=("c2",)),
            mock.patch.object(od, "_reference_root_move", return_value="b4"),
            mock.patch.object(
                od,
                "_run_multi_position_raw_nn_cached",
                return_value=(child_payloads, 0),
            ) as raw_nn_mock,
        ):
            payload, cache_hit = od._run_fair_root_candidate_sweep_cached(
                board_size=5,
                raw_nn_cache=raw_cache,
            )
            cached_payload, cached_hit = od._run_fair_root_candidate_sweep_cached(
                board_size=5,
                raw_nn_cache=raw_cache,
            )
        self.assertEqual(cache_hit, 0)
        self.assertEqual(cached_hit, 1)
        self.assertEqual(payload, {"m": [["c2", 0.25], ["b4", 0.75]]})
        self.assertEqual(cached_payload, payload)
        raw_nn_mock.assert_called_once()
        self.assertEqual(
            raw_nn_mock.call_args.kwargs["position_inputs"],
            [od.lps._position_after_move(root_position, move) for move in requested_moves],
        )

    def test_apply_prior_weighted_graph_values_propagates_retained_children(self):
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
        od._apply_prior_weighted_graph_values(nodes=nodes)
        root_rows = {row["move"]: row for row in nodes[0]["candidates"]}
        child_rows = {row["move"]: row for row in nodes[1]["candidates"]}
        self.assertIsNone(nodes[0]["tree_red_winrate"])
        self.assertAlmostEqual(nodes[1]["tree_red_winrate"], 0.7, places=6)
        self.assertAlmostEqual(root_rows["a3"]["tree_mover_winrate"], 0.7, places=6)
        self.assertAlmostEqual(root_rows["b4"]["tree_mover_winrate"], 0.61, places=6)
        self.assertAlmostEqual(child_rows["d8"]["tree_mover_winrate"], 0.2, places=6)
        self.assertAlmostEqual(child_rows["f7"]["tree_mover_winrate"], 0.6, places=6)
        self.assertAlmostEqual(child_rows["g8"]["tree_mover_winrate"], 0.3, places=6)


    def test_prune_raw_nn_cache_keeps_retained_and_nonretained_child_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_path = tmp_path / "openings-s3.json"
            cache_path = tmp_path / "cache.json"
            root_position = od._empty_position(board_size=3)
            retained_child = od.lps._position_after_move(root_position, "a1")
            nonretained_child = od.lps._position_after_move(root_position, "b1")
            payload = {
                "nodes": [
                    {
                        "parent": None,
                        "move": None,
                        "candidates": [
                            {
                                "move": "a1",
                                "rank": 1,
                                "prior": 0.4,
                                "stone_fraction": 1.0,
                                "candidate_weight": 1.0,
                                "importance": 0.9,
                                "child": 1,
                                "tree_mover_winrate": 0.6,
                            }
                        ],
                        "nonretained_candidates": [{"move": "b1", "importance": 0.7}],
                    },
                    {
                        "parent": 0,
                        "move": "a1",
                        "candidates": [],
                    },
                ],
            }
            od.aj.dump_tree(output_path, payload)
            cache = {
                od.lps._cache_key(root_position): {"r": 0.5},
                od.lps._cache_key(retained_child): {"r": 0.6},
                od.lps._cache_key(nonretained_child): {"r": 0.4},
                "unused": {"r": 0.1},
            }
            od.lps._save_raw_nn_cache(
                cache_path,
                {key: od.lps._encode_compact_raw_nn_payload(value) for key, value in cache.items()},
            )

            with mock.patch.object(od, "_raw_nn_cache_path", return_value=cache_path):
                _backup_path, _cache_path, before, after, missing_nodes, missing_children = od._prune_raw_nn_cache(
                    board_size=3,
                    output_path=output_path,
                )

            pruned = od.lps._load_raw_nn_cache(cache_path)
            self.assertEqual(before, 4)
            self.assertEqual(after, 3)
            self.assertEqual(missing_nodes, 0)
            self.assertEqual(missing_children, 0)
            self.assertIn(od.lps._cache_key(root_position), pruned)
            self.assertIn(od.lps._cache_key(retained_child), pruned)
            self.assertIn(od.lps._cache_key(nonretained_child), pruned)
            self.assertNotIn("unused", pruned)


if __name__ == "__main__":
    unittest.main()

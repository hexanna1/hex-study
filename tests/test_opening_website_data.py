from pathlib import Path
import tempfile
import unittest

import artifact_json as aj
import opening_website_data as owd


def _candidate(move, *, prior, child, tree_mover_winrate):
    return {
        "move": move,
        "rank": 1,
        "prior": prior,
        "stone_fraction": 1.0,
        "candidate_weight": 1.0,
        "importance": 0.4,
        "child": child,
        "tree_mover_winrate": tree_mover_winrate,
    }


def _node(parent, move, ply, importance, candidates, tree_red_winrate=None):
    return {
        "parent": parent,
        "move": move,
        "ply": ply,
        "importance": importance,
        "tree_red_winrate": tree_red_winrate,
        "candidates": candidates,
    }


class OpeningWebsiteDataTests(unittest.TestCase):
    def _artifact(self):
        return {
            "board_size": 3,
            "root": 0,
            "root_openings": ["a1", "b1"],
            "root_study": None,
            "completed": True,
            "completed_ply": 5,
            "nodes": [
                _node(
                    None,
                    None,
                    0,
                    1.0,
                    [
                        _candidate("a1", prior=0.5, child=1, tree_mover_winrate=0.65),
                        _candidate("b1", prior=0.5, child=2, tree_mover_winrate=0.65),
                    ],
                ),
                _node(0, "a1", 1, 0.2, [_candidate("c1", prior=1.0, child=3, tree_mover_winrate=0.35)], 0.65),
                _node(0, "b1", 1, 0.3, [_candidate("c2", prior=1.0, child=4, tree_mover_winrate=0.35)], 0.65),
                _node(1, "c1", 2, 0.2, [_candidate("b1", prior=1.0, child=5, tree_mover_winrate=0.65)], 0.65),
                _node(2, "c2", 2, 0.3, [_candidate("a1", prior=1.0, child=6, tree_mover_winrate=0.65)], 0.65),
                _node(3, "b1", 3, 0.2, [_candidate("c2", prior=1.0, child=7, tree_mover_winrate=0.35)], 0.65),
                _node(4, "a1", 3, 0.3, [_candidate("c1", prior=1.0, child=7, tree_mover_winrate=0.35)], 0.65),
                _node(
                    5,
                    "c2",
                    4,
                    0.4,
                    [
                        _candidate("b2", prior=0.75, child=8, tree_mover_winrate=0.70),
                        _candidate("a2", prior=0.25, child=9, tree_mover_winrate=0.50),
                    ],
                    tree_red_winrate=0.65,
                ),
                _node(7, "b2", 5, 0.1, [], tree_red_winrate=0.70),
                _node(7, "a2", 5, 0.25, [], tree_red_winrate=0.50),
            ],
        }

    def test_opening_graph_uses_one_redirect_and_absolute_root_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openings-s3.json"
            aj.dump_tree(path, self._artifact())
            raw = aj.load(path)
            bundle = owd.build_opening_bundle(artifacts_root=Path(tmp), board_size=3)

        root, graph = owd._opening_graph(nodes_raw=raw["n"], board_size=3)
        nodes, candidates, redirect_flags, redirect_targets = owd._linearize_opening_graph(
            root_node=root,
            graph=graph,
        )

        self.assertEqual(len(nodes), len(graph))
        self.assertEqual(graph[5]["candidates"][0]["target"], 7)
        self.assertEqual(graph[6]["candidates"][0]["target"], 7)
        self.assertEqual(sum(redirect_flags), 1)
        self.assertEqual(len(redirect_targets), 1)
        self.assertEqual(owd.HEADER_STRUCT.unpack_from(bundle)[4], 1)
        root_candidates = candidates[:nodes[0]["candidate_count"]]
        self.assertTrue(
            all(candidate["parent_red_winrate"] is None for candidate in root_candidates)
        )


if __name__ == "__main__":
    unittest.main()

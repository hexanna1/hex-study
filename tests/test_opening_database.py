import unittest

import opening_database as od


def _candidate(move, winrate, *, prior=None, child=None, retained=True):
    return {
        "move": move, "raw_mover_winrate": winrate, "prior": prior,
        "child": child, "retained": retained,
    }


class OpeningDatabaseTests(unittest.TestCase):
    def test_graph_values_use_retained_prior_mass_and_convert_side_at_each_ply(self):
        nodes = [
            {"ply": 0, "candidates": [
                _candidate("a3", 0.74, child=1), _candidate("b4", 0.61, retained=False),
            ]},
            {"ply": 1, "candidates": [
                _candidate("d8", 0.2, prior=0.75), _candidate("f7", 0.6, prior=0.25),
                _candidate("g8", 0.3, prior=0.1, retained=False),
            ]},
        ]
        od._apply_prior_weighted_graph_values(nodes=nodes)
        self.assertIsNone(nodes[0]["tree_red_winrate"])
        self.assertAlmostEqual(nodes[1]["tree_red_winrate"], 0.7)
        self.assertAlmostEqual(nodes[0]["candidates"][0]["tree_mover_winrate"], 0.7)
        self.assertAlmostEqual(nodes[0]["candidates"][1]["tree_mover_winrate"], 0.61)
        self.assertEqual([row["tree_mover_winrate"] for row in nodes[1]["candidates"]], [0.2, 0.6, 0.3])

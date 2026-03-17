import unittest

import study_common as lps
from local_pattern_representative import (
    build_study_spec,
    extract_pattern,
    generate_representatives,
    parse_balance_profiles,
    point_to_cell,
)


class LocalPatternRepresentativeTests(unittest.TestCase):
    def test_extract_pattern_side_consistent_plus_minus(self):
        ex = extract_pattern("https://hexworld.org/board/#14c1,h7h8i7h9g8")
        self.assertEqual(ex.board_size_source, 14)
        self.assertEqual(ex.to_play_at_cursor, "blue")
        self.assertEqual(set(ex.plus_cells), {"h8", "h9"})
        self.assertEqual(set(ex.minus_cells), {"h7", "i7", "g8"})
        self.assertEqual(len(ex.source_cells_all), 5)


    def test_generate_representatives_with_balance_and_dedup(self):
        ex = extract_pattern("https://hexworld.org/board/#14c1,h7h8i7h9g8")
        balances = parse_balance_profiles(["c2,d5"])
        reps = generate_representatives(
            extracted=ex,
            board_size=19,
            symmetry="d6",
            balance_profiles=balances,
        )
        self.assertGreaterEqual(len(reps), 1)
        self.assertEqual(len({r.position for r in reps}), len(reps))
        for r in reps:
            self.assertTrue(r.position.startswith("https://hexworld.org/board/#19c1,"))
            # balance stones present in final occupancy
            self.assertIn((3, 2), set(r.red_cells) | set(r.blue_cells))
            self.assertIn((4, 5), set(r.red_cells) | set(r.blue_cells))





    def test_Δ_candidates_threshold_includes_local_and_excludes_far_cells(self):
        ex = extract_pattern("https://hexworld.org/board/#4c1,a1")
        reps = generate_representatives(
            extracted=ex,
            board_size=7,
            symmetry="identity",
            balance_profiles=parse_balance_profiles(["none"]),
        )
        spec = build_study_spec(
            extracted=ex,
            representatives=reps,
            board_size=7,
            symmetry="identity",
            candidate_mode="auto-near-pattern",
            explicit_candidates=[],
            candidate_Δ_max=12,
            search_seconds=1.0,
            awrn=None,
        )
        cands = spec["experiments"][0]["candidates"]
        self.assertIn("f5", cands)  # local to centered pattern
        self.assertNotIn("a1", cands)  # far from centered pattern, Δ > 12

    def test_build_study_spec_emits_nontrivial_local_key_orbit_for_symmetric_pattern(self):
        ex = extract_pattern("https://hexworld.org/board/#4c1,a1")
        reps = generate_representatives(
            extracted=ex,
            board_size=7,
            symmetry="d6",
            balance_profiles=parse_balance_profiles(["none"]),
        )
        spec = build_study_spec(
            extracted=ex,
            representatives=reps,
            board_size=7,
            symmetry="d6",
            candidate_mode="auto-near-pattern",
            explicit_candidates=[],
            candidate_Δ_max=12,
            search_seconds=1.0,
            awrn=None,
        )
        exp_name = spec["experiments"][0]["name"]
        md = spec["generator_meta"]["experiment_meta"][exp_name]
        orbit = md["local_key_orbit"]
        self.assertEqual(len(orbit), 12)
        self.assertIn({"transform_id": 0, "norm_shift": [0, 0]}, orbit)


    def test_generate_representatives_edge_bilateral_yields_two_orientations(self):
        ex = extract_pattern("https://hexworld.org/board/#21c1,k11l10")
        reps = generate_representatives(
            extracted=ex,
            board_size=21,
            symmetry="edge-bilateral",
            balance_profiles=parse_balance_profiles(["a1,d2"]),
            placement="edge",
            edge_anchor_col_from_right=1,
        )
        self.assertEqual(len(reps), 2)
        self.assertEqual({rep.orientation_transform_id for rep in reps}, {0, 6})

        spec = build_study_spec(
            extracted=ex,
            representatives=reps,
            board_size=21,
            symmetry="edge-bilateral",
            candidate_mode="auto-near-pattern",
            explicit_candidates=[],
            candidate_Δ_max=7,
        )
        exp_meta = spec["generator_meta"]["experiment_meta"]
        key_1 = lps._candidate_key_local_for_move(point_to_cell(*reps[0].plus_abs[0]), exp_meta["b01-o01"])
        key_2 = lps._candidate_key_local_for_move(point_to_cell(*reps[1].plus_abs[0]), exp_meta["b01-o02"])
        self.assertEqual(key_1, "0,0")
        self.assertEqual(key_2, "0,0")

if __name__ == "__main__":
    unittest.main()

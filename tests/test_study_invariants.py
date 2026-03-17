import unittest

from hex_symmetry import apply_transform_ax
import hex_board
import native_tools
import pattern_scoring


class StudyInvariantTests(unittest.TestCase):
    def test_native_normalization_respects_triangle_orientation_and_side_to_move(self):
        rows = native_tools.normalize_moves(
            requests=[
                ("https://hexworld.org/board/#11c1,a3f6b3g6", ["a2", "b2", "c2"]),
                ("https://hexworld.org/board/#11c1,k9f6j9g6", ["k10", "j10", "i10"]),
                ("https://hexworld.org/board/#11c1,f6c1g6c2h6", ["b1", "b2", "b3"]),
                ("https://hexworld.org/board/#11c1,f6i3g6", ["j3", "j2"]),
                ("https://hexworld.org/board/#11c1,a3", ["a2", "b2"]),
            ],
            board_size=11,
        )
        self.assertEqual(rows, [["c2"] * 3, ["i10"] * 3, ["b3"] * 3, ["j2"] * 2, ["a2", "b2"]])

    def test_position_cache_identity_uses_replayed_board_and_cursor(self):
        self.assertEqual(
            hex_board.position_cache_key("21c1,a1d2k11l11"),
            hex_board.position_cache_key("21c1,k11d2a1l11"),
        )
        # Swap transposes the opening stone. Future moves do not affect the cursor.
        for source, expected in (
            ("5c1,b3:s,a1", (5, set(), {(3, 2)}, "red")),
            ("5c1,a1:p,b3", (5, {(1, 1)}, set(), "red")),
            ("5c1,a1", (5, {(1, 1)}, set(), "blue")),
        ):
            with self.subTest(source=source):
                size, red, blue, side = hex_board.position_state(source)
                self.assertEqual((size, red, blue, side), expected)
                serialized = hex_board.serialize_position(
                    board_size=size, red_cells=tuple(red), blue_cells=tuple(blue), to_play=side,
                )
                self.assertEqual(hex_board.position_state(serialized), expected)
                self.assertEqual(hex_board.position_cache_key(serialized), hex_board.position_cache_key(source))

    def test_candidate_keys_round_trip_all_frames_and_identify_stabilizer_orbits(self):
        base = (2, -1)
        shift, offset = (1, -2), (10, 10)
        for transform in range(12):
            q, r = apply_transform_ax(base, transform)
            move = hex_board.point_to_cell(q - shift[0] + offset[0], r - shift[1] + offset[1])
            metadata = {
                "orientation_transform_id": transform,
                "orientation_norm_shift": list(shift),
                "board_offset": list(offset),
            }
            self.assertEqual(pattern_scoring.candidate_key(move, metadata), "2,-1")

        metadata = {
            "orientation_transform_id": 0, "orientation_norm_shift": [0, 0], "board_offset": [10, 10],
            "local_key_orbit": [
                {"transform_id": 0, "norm_shift": [0, 0]},
                {"transform_id": 3, "norm_shift": [-1, 0]},
                {"transform_id": 8, "norm_shift": [-1, 0]},
                {"transform_id": 11, "norm_shift": [0, 0]},
            ],
        }
        self.assertEqual(pattern_scoring.candidate_key("j11", metadata), "0,1")
        self.assertEqual(pattern_scoring.candidate_key("k9", metadata), "0,1")

    def test_pooling_precedes_normalization(self):
        # Normalizing each experiment first would reverse the winning candidate.
        rows = [
            {"experiment": experiment, "candidate_key_local": key, "corrected_value": value}
            for experiment, values in (("e1", (100.0, 20.0)), ("e2", (1.0, 50.0)))
            for key, value in zip(("a", "b"), values)
        ]
        pooled = pattern_scoring.pool_candidates(rows, total_representatives=2)
        self.assertEqual([row["candidate_key_local"] for row in pooled], ["a", "b"])
        self.assertAlmostEqual(pooled[0]["mean_stone_fraction"], 1.0)
        self.assertAlmostEqual(pooled[1]["mean_stone_fraction"], 70.0 / 101.0)

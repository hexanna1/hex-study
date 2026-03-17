from pathlib import Path
import unittest
from unittest import mock

import study_common


class StudyCommonTests(unittest.TestCase):
    def test_native_raw_nn_batch_uses_canonical_positions_and_decodes_compact_output(self):
        position = "https://hexworld.org/board/#5c1,a1b1"
        completed = mock.Mock(
            returncode=0,
            stdout='{"position":"5,a1b1","r":250000,"m":[["c3",500000],["pass",125000]]}\n',
            stderr="",
        )
        with mock.patch.object(study_common.subprocess, "run", return_value=completed) as run_mock:
            payloads = study_common._run_multi_position_raw_nn_native(
                position_inputs=[position],
                board_size=5,
                move_limit=24,
            )
        self.assertEqual(
            payloads[position],
            {"r": 0.25, "m": [["c3", 0.5], ["pass", 0.125]]},
        )
        self.assertEqual(run_mock.call_args.kwargs["input"], "5,a1b1\n")
        command = run_mock.call_args.args[0]
        self.assertIn("batchrawnn", command)
        self.assertEqual(command[command.index("-board-size") + 1], "5")
        self.assertEqual(command[command.index("-top-n") + 1], "24")

    def test_native_raw_nn_cached_checkpoints_once_per_chunk(self):
        positions = ["5,a1b1", "5,a1b1c1d1", "5,a2b2c2d2"]
        fetched = {position: {"r": 0.5, "m": [["c3", 0.25]]} for position in positions}

        def run_batch(*, position_inputs, **_kwargs):
            return {position: fetched[position] for position in position_inputs}

        cache: dict = {}
        with (
            mock.patch.object(study_common, "_run_multi_position_raw_nn_native", side_effect=run_batch) as run_mock,
            mock.patch.object(study_common, "_save_raw_nn_cache") as save_mock,
        ):
            payloads, cache_hits = study_common._ensure_raw_nn_cache_entries(
                position_inputs=positions,
                raw_nn_cache=cache,
                board_size=5,
                raw_nn_cache_path=Path("cache.json"),
                chunk_size=2,
                move_limit=24,
            )
        self.assertEqual(
            payloads,
            {position: study_common._encode_compact_raw_nn_payload(payload) for position, payload in fetched.items()},
        )
        self.assertEqual(cache_hits, 0)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(save_mock.call_count, 2)

    def test_native_raw_nn_batch_surfaces_position_error(self):
        completed = mock.Mock(
            returncode=1,
            stdout='{"position":"5,a1","error":"Synthetic evaluation failure for 5,a1"}\n',
            stderr="",
        )
        with mock.patch.object(study_common.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(
                ValueError,
                r"Synthetic evaluation failure for 5,a1",
            ):
                study_common._run_multi_position_raw_nn_native(
                    position_inputs=["5,a1"],
                    board_size=5,
                    move_limit=24,
                )

    def test_candidate_key_local_roundtrip_for_multiple_transforms(self):
        base_rel = (2, -1)
        offset = (10, 10)
        for transform_id in (0, 1, 7):
            transformed = study_common._apply_transform_ax(base_rel, transform_id)
            abs_col = offset[0] + transformed[0]
            abs_row = offset[1] + transformed[1]
            move = f"{study_common._letters_for_col(abs_col)}{abs_row}"
            exp_meta = {
                "orientation_transform_id": transform_id,
                "orientation_norm_shift": [0, 0],
                "placement_offset": [offset[0], offset[1]],
            }
            key = study_common._candidate_key_local_for_move(move, exp_meta)
            self.assertEqual(key, f"{base_rel[0]},{base_rel[1]}")

    def test_candidate_key_local_orbit_canonicalizes_symmetric_geometric_keys(self):
        exp_meta_base = {
            "orientation_transform_id": 1,
            "orientation_norm_shift": [0, 0],
            "placement_offset": [10, 10],
        }

        def move_for_base_rel(base_rel: tuple[int, int], exp_meta: dict) -> str:
            transform_id = int(exp_meta["orientation_transform_id"])
            shift = exp_meta["orientation_norm_shift"]
            offset = exp_meta["placement_offset"]
            transformed = study_common._apply_transform_ax(base_rel, transform_id)
            col = int(offset[0]) + transformed[0] - int(shift[0])
            row = int(offset[1]) + transformed[1] - int(shift[1])
            return f"{study_common._letters_for_col(col)}{row}"

        move_a = move_for_base_rel((0, 1), exp_meta_base)
        move_b = move_for_base_rel((1, -1), exp_meta_base)

        key_a_raw = study_common._candidate_key_local_for_move(move_a, exp_meta_base)
        key_b_raw = study_common._candidate_key_local_for_move(move_b, exp_meta_base)
        self.assertEqual(key_a_raw, "0,1")
        self.assertEqual(key_b_raw, "1,-1")

        exp_meta_orbit = {
            **exp_meta_base,
            "local_key_orbit": [
                {"transform_id": 0, "norm_shift": [0, 0]},
                {"transform_id": 3, "norm_shift": [-1, 0]},
                {"transform_id": 8, "norm_shift": [-1, 0]},
                {"transform_id": 11, "norm_shift": [0, 0]},
            ],
        }
        key_a = study_common._candidate_key_local_for_move(move_a, exp_meta_orbit)
        key_b = study_common._candidate_key_local_for_move(move_b, exp_meta_orbit)
        self.assertEqual(key_a, "0,1")
        self.assertEqual(key_b, "0,1")

    def test_build_pooled_candidates_normalizes_after_pooling(self):
        rows = [
            {
                "experiment": "e1",
                "candidate_key_local": "1,0",
                "candidate_abs": "m10",
                "corrected_value": 0.8,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "1,0",
                "candidate_abs": "n9",
                "corrected_value": 0.6,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "2,0",
                "candidate_abs": "n10",
                "corrected_value": 0.3,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "2,0",
                "candidate_abs": "o9",
                "corrected_value": 0.5,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "pass_proxy",
                "candidate_abs": "m1",
                "corrected_value": 0.0,
            },
        ]
        pooled = study_common._build_pooled_candidates(rows, total_representatives=2)
        self.assertEqual(len(pooled), 3)
        by_key = {r["candidate_key_local"]: r for r in pooled}
        self.assertAlmostEqual(by_key["1,0"]["mean_corrected_value"], 0.7, places=6)
        self.assertAlmostEqual(by_key["2,0"]["mean_corrected_value"], 0.4, places=6)
        self.assertAlmostEqual(by_key["1,0"]["mean_stone_fraction"], 1.0, places=6)
        self.assertAlmostEqual(by_key["2,0"]["mean_stone_fraction"], 4.0 / 7.0, places=6)
        self.assertAlmostEqual(by_key["pass_proxy"]["mean_stone_fraction"], 0.0, places=6)

    def test_build_pooled_candidates_all_zero_when_pass_is_best_on_average(self):
        rows = [
            {
                "experiment": "e1",
                "candidate_key_local": "1,0",
                "candidate_abs": "m10",
                "corrected_value": -0.1,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "1,0",
                "candidate_abs": "n9",
                "corrected_value": -0.2,
            },
            {
                "experiment": "e1",
                "candidate_key_local": "pass_proxy",
                "candidate_abs": "m1",
                "corrected_value": 0.0,
            },
            {
                "experiment": "e2",
                "candidate_key_local": "pass_proxy",
                "candidate_abs": "m1",
                "corrected_value": 0.0,
            },
        ]
        pooled = study_common._build_pooled_candidates(
            rows,
            total_representatives=2,
            value_field="corrected_value",
        )
        by_key = {r["candidate_key_local"]: r for r in pooled}
        self.assertAlmostEqual(by_key["pass_proxy"]["mean_corrected_value"], 0.0, places=6)
        self.assertAlmostEqual(by_key["pass_proxy"]["mean_stone_fraction"], 0.0, places=6)
        self.assertAlmostEqual(by_key["1,0"]["mean_stone_fraction"], 0.0, places=6)

if __name__ == "__main__":
    unittest.main()

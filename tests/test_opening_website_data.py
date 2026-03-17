from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import opening_website_data as owd
try:
    from bundle_test_utils import read_little_endian_bits
except ModuleNotFoundError:  # pragma: no cover - package-style unittest invocation
    from tests.bundle_test_utils import read_little_endian_bits


class OpeningWebsiteDataTests(unittest.TestCase):
    def _write_artifact_payload(self, path: Path, payload: dict[str, object]) -> None:
        payload = dict(payload)
        payload.setdefault("version", 1)
        payload.setdefault("format", "move_tree")
        payload.setdefault("mode", "openings")
        payload.setdefault("root", 0)
        payload.setdefault("root_study", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _packed_move_id_bits(self, board_size: int) -> int:
        return ((int(board_size) * int(board_size)) - 1).bit_length()

    def _read_bits(self, payload: bytes, *, offset: int, bit_offset: int, bits: int) -> int:
        return read_little_endian_bits(payload, offset=offset, bit_offset=bit_offset, bits=bits, chunk_bytes=6)

    def _decode_bundle(self, payload: bytes) -> dict[str, object]:
        magic, version, board_size, node_count, candidate_count = owd.HEADER_STRUCT.unpack_from(payload, 0)
        self.assertEqual(magic, owd.BUNDLE_MAGIC)
        nodes = []
        offset = owd.HEADER_STRUCT.size
        for _ in range(node_count):
            word = int.from_bytes(payload[offset:offset + owd.NODE_ROW_SIZE], "little")
            nodes.append(
                [
                    bool((word >> owd.PACKED_NODE_IS_CORE_SHIFT) & 1),
                    word & owd.PACKED_NODE_COUNT_MASK,
                    bool((word >> owd.PACKED_NODE_HAS_CHILDREN_SHIFT) & 1),
                ]
            )
            offset += owd.NODE_ROW_SIZE
        move_id_bits = self._packed_move_id_bits(board_size)
        move_high_bits = max(0, move_id_bits - 8)
        move_low_offset = offset
        move_high_offset = move_low_offset + candidate_count
        prior_low_offset = move_high_offset + ((candidate_count * move_high_bits + 7) // 8)
        prior_high_offset = prior_low_offset + ((candidate_count * 2 + 7) // 8)
        delta_stream_offset = prior_high_offset + candidate_count
        delta_bit_length = sum(count * owd.PACKED_CANDIDATE_DELTA_BITS for _is_core, count, _has_children in nodes if count > 1)
        first_prior_count = sum(1 for _is_core, count, _has_children in nodes if count > 0)
        exception_offset = delta_stream_offset + ((delta_bit_length + 7) // 8)
        candidates = []
        node_offset = 0
        candidate_index = 0
        first_prior_index = 0
        drop_prior_index = 0
        delta_bit_offset = 0
        exception_index = 0

        def read_exception() -> int:
            nonlocal exception_index
            value = self._read_bits(
                payload,
                offset=exception_offset,
                bit_offset=exception_index * owd.PACKED_CANDIDATE_METRIC_BITS,
                bits=owd.PACKED_CANDIDATE_METRIC_BITS,
            )
            exception_index += 1
            return value

        def mover_from_red(red: int | None, ply: int) -> int:
            if red is None:
                return 1023
            return int(red) if int(ply) % 2 == 0 else 1000 - int(red)

        def walk(parent_edge_red: int | None, ply: int) -> None:
            nonlocal node_offset, candidate_index, first_prior_index, drop_prior_index, delta_bit_offset
            _is_core, count, has_child = nodes[node_offset]
            node_offset += 1
            children: list[int | None] = []
            previous_prior: int | None = None
            for idx in range(count):
                row_index = candidate_index
                move_low = payload[move_low_offset + row_index]
                move_high = (
                    self._read_bits(
                        payload,
                        offset=move_high_offset,
                        bit_offset=row_index * move_high_bits,
                        bits=move_high_bits,
                    )
                    if move_high_bits
                    else 0
                )
                move_id = move_low + (move_high << 8)
                if idx == 0:
                    prior_index = first_prior_index
                    first_prior_index += 1
                else:
                    prior_index = first_prior_count + drop_prior_index
                    drop_prior_index += 1
                prior_value = self._read_bits(
                    payload,
                    offset=prior_low_offset,
                    bit_offset=prior_index * 2,
                    bits=2,
                ) + (payload[prior_high_offset + prior_index] << 2)
                prior = prior_value if idx == 0 else int(previous_prior) - prior_value
                previous_prior = prior
                if count == 1:
                    red = parent_edge_red
                else:
                    delta = self._read_bits(
                        payload,
                        offset=delta_stream_offset,
                        bit_offset=delta_bit_offset,
                        bits=owd.PACKED_CANDIDATE_DELTA_BITS,
                    )
                    delta_bit_offset += owd.PACKED_CANDIDATE_DELTA_BITS
                    if delta == owd.PACKED_CANDIDATE_DELTA_ESCAPE or parent_edge_red is None:
                        red = read_exception()
                    else:
                        red = int(parent_edge_red) + int(delta) - owd.PACKED_CANDIDATE_DELTA_MAX_ABS
                candidates.append([move_id, prior, mover_from_red(red, ply), has_child])
                candidate_index += 1
                if has_child:
                    children.append(red)
            for child_red in children:
                walk(child_red, ply + 1)

        walk(None, 0)
        offset = exception_offset + ((exception_index * owd.PACKED_CANDIDATE_METRIC_BITS + 7) // 8)
        self.assertEqual(node_offset, len(nodes))
        self.assertEqual(candidate_index, candidate_count)
        self.assertEqual(first_prior_index, first_prior_count)
        self.assertEqual(drop_prior_index, candidate_count - first_prior_count)
        self.assertEqual(delta_bit_offset, delta_bit_length)
        self.assertEqual(offset, len(payload))
        return {
            "version": version,
            "board_size": board_size,
            "nodes": nodes,
            "candidates": candidates,
        }

    def _write_artifact(
        self,
        *,
        artifacts_root: Path,
        board_size: int,
        root_openings: list[str],
        completed: bool = False,
        completed_ply: int = 1,
    ) -> None:
        artifact_path = artifacts_root / f"openings-s{board_size}.json"
        self._write_artifact_payload(
            artifact_path,
            {
                "board_size": board_size,
                "root_openings": root_openings,
                "completed": completed,
                "completed_ply": completed_ply,
                "nodes": [
                    {
                        "parent": None,
                        "move": None,
                        "ply": 0,
                        "importance": 1.0,
                        "candidates": [
                            {
                                "move": root_openings[0],
                                "child": None,
                                "prior": None,
                                "tree_mover_winrate": None,
                            }
                        ],
                    }
                ],
            },
        )

    def test_build_opening_bundle_compacts_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            artifacts_root = repo_root / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            self._write_artifact_payload(
                artifact_path,
                {
                    "board_size": 11,
                    "root_openings": ["a3", "a4"],
                    "completed": True,
                    "completed_ply": 3,
                    "nodes": [
                        {
                            "parent": None,
                            "move": None,
                            "ply": 0,
                            "importance": 1.0,
                            "candidates": [
                                {
                                    "move": "a3",
                                    "child": 1,
                                    "prior": None,
                                    "tree_mover_winrate": 0.71,
                                },
                                {
                                    "move": "b4",
                                    "child": 3,
                                    "prior": 0.25,
                                    "tree_mover_winrate": 0.5,
                                },
                            ],
                        },
                        {
                            "parent": 0,
                            "move": "a3",
                            "ply": 1,
                            "importance": 0.87654,
                            "candidates": [
                                {
                                    "move": "d6",
                                    "child": 2,
                                    "prior": 0.4444444,
                                    "tree_mover_winrate": 0.29,
                                },
                            ],
                            "nonretained_candidates": [{"move": "c5", "importance": 0.7}],
                        },
                        {
                            "parent": 1,
                            "move": "d6",
                            "ply": 2,
                            "importance": 0.81234,
                            "candidates": [],
                        },
                        {
                            "parent": 0,
                            "move": "b4",
                            "ply": 1,
                            "importance": 0.71234,
                            "candidates": [],
                        },
                    ],
                },
            )

            payload = owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)
            decoded = self._decode_bundle(payload)

        self.assertEqual(decoded["version"], 1)
        self.assertEqual(decoded["board_size"], 11)
        self.assertEqual(
            decoded["nodes"],
            [
                [True, 2, True],
                [False, 1, True],
                [False, 0, False],
                [False, 0, False],
            ],
        )
        self.assertEqual(
            decoded["candidates"],
            [
                [22, 1023, 710, True],
                [34, 250, 500, True],
                [58, 444, 290, True],
            ],
        )

    def test_write_opening_bundles_maps_board_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts_root = root / "artifacts"
            out_path = root / "current.json"
            self._write_artifact(artifacts_root=artifacts_root, board_size=11, root_openings=["a3"])
            self._write_artifact(artifacts_root=artifacts_root, board_size=12, root_openings=["b4"])
            owd.write_opening_bundles(artifacts_root=artifacts_root, out_path=out_path, board_sizes=[11, 12])
            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            written_sizes = {
                int(size): self._decode_bundle((out_path.parent / name).read_bytes())["board_size"]
                for size, name in manifest["bundles"].items()
            }
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(written_sizes, {11: 11, 12: 12})
        self.assertTrue(all(name.startswith("opening_index.") for name in manifest["bundles"].values()))

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import opening_website_data as owd


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

    def _candidate_row_bits(self, board_size: int) -> int:
        return self._packed_move_id_bits(board_size) + owd.PACKED_CANDIDATE_METRIC_BITS + owd.PACKED_CANDIDATE_DELTA_BITS + 1

    def _single_candidate_row_bits(self, board_size: int) -> int:
        return self._packed_move_id_bits(board_size) + owd.PACKED_CANDIDATE_METRIC_BITS + 1

    def _read_bits(self, payload: bytes, *, offset: int, bit_offset: int, bits: int) -> int:
        byte_offset = offset + (int(bit_offset) // 8)
        shift = int(bit_offset) % 8
        chunk = int.from_bytes(payload[byte_offset:byte_offset + 6], "little")
        return (chunk >> shift) & ((1 << int(bits)) - 1)

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
                ]
            )
            offset += owd.NODE_ROW_SIZE
        move_id_bits = self._packed_move_id_bits(board_size)
        row_bits = self._candidate_row_bits(board_size)
        single_row_bits = self._single_candidate_row_bits(board_size)
        candidate_stream_offset = offset
        candidate_bit_length = sum(count * (single_row_bits if count == 1 else row_bits) for _is_core, count in nodes)
        exception_offset = candidate_stream_offset + ((candidate_bit_length + 7) // 8)
        candidates = []
        node_offset = 0
        candidate_index = 0
        candidate_bit_offset = 0
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
                return owd.PACKED_OPTIONAL_NULL
            return int(red) if int(ply) % 2 == 0 else 1000 - int(red)

        def walk(parent_edge_red: int | None, ply: int) -> None:
            nonlocal node_offset, candidate_index, candidate_bit_offset
            _is_core, count = nodes[node_offset]
            node_offset += 1
            children: list[int | None] = []
            for _ in range(count):
                bits = single_row_bits if count == 1 else row_bits
                word = self._read_bits(
                    payload,
                    offset=candidate_stream_offset,
                    bit_offset=candidate_bit_offset,
                    bits=bits,
                )
                candidate_bit_offset += bits
                move_id = word & ((1 << move_id_bits) - 1)
                prior = (word >> move_id_bits) & 0x3FF
                if count == 1:
                    red = parent_edge_red
                    has_child = bool((word >> (move_id_bits + owd.PACKED_CANDIDATE_METRIC_BITS)) & 1)
                else:
                    delta = (word >> (move_id_bits + owd.PACKED_CANDIDATE_METRIC_BITS)) & owd.PACKED_CANDIDATE_DELTA_ESCAPE
                    has_child = bool((word >> (move_id_bits + owd.PACKED_CANDIDATE_METRIC_BITS + owd.PACKED_CANDIDATE_DELTA_BITS)) & 1)
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
        self.assertEqual(candidate_bit_offset, candidate_bit_length)
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
                                    "child": None,
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
                [True, 2],
                [False, 1],
                [False, 0],
            ],
        )
        self.assertEqual(
            decoded["candidates"],
            [
                [22, 1023, 710, True],
                [34, 250, 500, False],
                [58, 444, 290, True],
            ],
        )

    def test_build_opening_bundle_keeps_frontier_candidates_without_child_node(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            self._write_artifact_payload(
                artifact_path,
                {
                    "board_size": 11,
                    "root_openings": ["a3"],
                    "completed": True,
                    "completed_ply": 1,
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
                                    "prior": 0.5,
                                    "tree_mover_winrate": None,
                                },
                            ],
                        },
                    ],
                },
            )

            payload = owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)
            decoded = self._decode_bundle(payload)

        self.assertEqual(decoded["nodes"], [[True, 1]])
        self.assertEqual(decoded["candidates"], [[22, 500, 1023, False]])

    def test_write_opening_bundles_writes_single_manifest_and_hashed_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            artifacts_root = repo_root / "artifacts" / "openings"
            out_path = repo_root / "docs" / "data" / "openings_current.json"
            self._write_artifact(artifacts_root=artifacts_root, board_size=11, root_openings=["a3"])
            self._write_artifact(artifacts_root=artifacts_root, board_size=12, root_openings=["b4"])

            owd.write_opening_bundles(artifacts_root=artifacts_root, out_path=out_path, board_sizes=[11, 12])

            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            bundle_11_path = out_path.parent / str(manifest["bundles"]["11"])
            bundle_12_path = out_path.parent / str(manifest["bundles"]["12"])
            bundle_11 = self._decode_bundle(bundle_11_path.read_bytes())
            bundle_12 = self._decode_bundle(bundle_12_path.read_bytes())

        self.assertEqual(manifest["version"], 1)
        self.assertRegex(str(manifest["bundles"]["11"]), r"^opening_index\.[0-9a-f]{12}\.bin$")
        self.assertRegex(str(manifest["bundles"]["12"]), r"^opening_index\.[0-9a-f]{12}\.bin$")
        self.assertEqual(bundle_11["board_size"], 11)
        self.assertEqual(bundle_12["board_size"], 12)

    def test_move_to_cell_id_handles_excel_style_columns(self) -> None:
        self.assertEqual(owd._move_to_cell_id("z1", board_size=28), 25)
        self.assertEqual(owd._move_to_cell_id("aa1", board_size=28), 26)
        self.assertEqual(owd._move_to_cell_id("ab1", board_size=28), 27)

    def test_build_opening_bundle_rejects_invalid_root_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            for root_node, expected_error in [
                (
                    {
                        "parent": 7,
                        "move": None,
                        "ply": 0,
                        "importance": 1.0,
                        "candidates": [],
                    },
                    "root parent must be null",
                ),
                (
                    {
                        "parent": None,
                        "move": "a3",
                        "ply": 0,
                        "importance": 1.0,
                        "candidates": [],
                    },
                    "root move must be empty",
                ),
            ]:
                with self.subTest(root_node=root_node):
                    self._write_artifact_payload(
                        artifact_path,
                        {
                            "board_size": 11,
                            "root_openings": ["a3"],
                            "completed": False,
                            "completed_ply": 1,
                            "nodes": [root_node],
                        },
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)

    def test_build_opening_bundle_rejects_non_root_with_null_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            self._write_artifact_payload(
                artifact_path,
                {
                    "board_size": 11,
                    "root_openings": ["a3"],
                    "completed": False,
                    "completed_ply": 2,
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
                                    "prior": 0.5,
                                    "tree_mover_winrate": 0.6,
                                },
                            ],
                        },
                        {
                            "parent": None,
                            "move": "a3",
                            "ply": 1,
                            "importance": 0.8,
                            "candidates": [],
                        },
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "non-root node with null parent"):
                owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)

if __name__ == "__main__":
    unittest.main()

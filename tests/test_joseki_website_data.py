from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joseki_website_data as jwd
try:
    from bundle_test_utils import read_little_endian_bits
except ModuleNotFoundError:  # pragma: no cover - package-style unittest invocation
    from tests.bundle_test_utils import read_little_endian_bits


class JosekiWebsiteDataTests(unittest.TestCase):
    def _read_bits(self, payload: bytes, *, offset: int, bit_offset: int, bits: int) -> int:
        return read_little_endian_bits(payload, offset=offset, bit_offset=bit_offset, bits=bits, chunk_bytes=4)

    def _decode_bundle(self, payload: bytes) -> dict[str, object]:
        magic, version, family_code, board_size, node_count, local_row_count = jwd.HEADER_STRUCT.unpack_from(payload, 0)
        self.assertEqual(magic, jwd.BUNDLE_MAGIC)
        nodes = []
        offset = jwd.HEADER_STRUCT.size
        node_tail_offset = offset
        local_rows_offset = node_tail_offset + ((node_count * jwd.PACKED_NODE_TAIL_BITS + 7) // 8)
        local_child_bits_offset = local_rows_offset + (local_row_count * jwd.LOCAL_ROW_STRUCT.size)
        for idx in range(node_count):
            word = self._read_bits(
                payload,
                offset=node_tail_offset,
                bit_offset=idx * jwd.PACKED_NODE_TAIL_BITS,
                bits=jwd.PACKED_NODE_TAIL_BITS,
            )
            local_count = word & ((1 << jwd.PACKED_NODE_LOCAL_COUNT_BITS) - 1)
            is_core = bool((word >> jwd.PACKED_NODE_IS_CORE_SHIFT) & 0x1)
            tenuki_retained = bool((word >> jwd.PACKED_NODE_TENUKI_RETAINED_SHIFT) & 0x1)
            tenuki_present = bool((word >> jwd.PACKED_NODE_TENUKI_PRESENT_SHIFT) & 0x1)
            tenuki_child = bool((word >> jwd.PACKED_NODE_TENUKI_CHILD_SHIFT) & 0x1)
            tenuki_sf = (word >> jwd.PACKED_NODE_TENUKI_SF_SHIFT) & 0x3FF
            nodes.append(
                [
                    local_count,
                    is_core,
                    tenuki_retained,
                    tenuki_present,
                    tenuki_child,
                    tenuki_sf,
                ]
            )
        local_rows = []
        for idx in range(local_row_count):
            row_offset = local_rows_offset + (idx * jwd.LOCAL_ROW_STRUCT.size)
            move_code, stone_fraction = jwd.LOCAL_ROW_STRUCT.unpack_from(payload, row_offset)
            local_rows.append([move_code, stone_fraction])
        local_child_bits = [
            bool(self._read_bits(payload, offset=local_child_bits_offset, bit_offset=idx, bits=1))
            for idx in range(local_row_count)
        ]
        offset = local_child_bits_offset + ((local_row_count + 7) // 8)
        self.assertEqual(offset, len(payload))
        return {
            "version": version,
            "family_code": family_code,
            "board_size": board_size,
            "nodes": nodes,
            "local_rows": local_rows,
            "local_child_bits": local_child_bits,
        }

    def _write_artifact(self, *, artifacts_root: Path, family: str, nodes: list[dict]) -> None:
        artifact_path = artifacts_root / f"joseki-{family.lower()}-s19.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "family": family,
                    "board_size": 19,
                    "nodes": nodes,
                }
            ),
            encoding="utf-8",
        )

    def test_build_family_bundle_compacts_parent_linked_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "joseki"
            self._write_artifact(
                artifacts_root=artifacts_root,
                family="A",
                nodes=[
                    {
                        "line": "",
                        "candidates": [
                            {"kind": "local", "local": [6, 5], "stone_fraction": 1.0},
                            {"kind": "local", "local": [5, 5], "stone_fraction": 0.5},
                        ],
                        "retained_lines": ["A[6,5]", "A[5,5]"],
                        "importance": 1.0,
                    },
                    {
                        "line": "A[6,5]",
                        "candidates": [
                            {"kind": "tenuki", "stone_fraction": 1.0},
                            {"kind": "local", "local": [4, 5], "stone_fraction": 0.977},
                        ],
                        "retained_lines": ["A[6,5:4,5]", "A[6,5:]"],
                        "importance": 0.99,
                    },
                    {
                        "line": "A[6,5:4,5]",
                        "candidates": [],
                        "retained_lines": [],
                        "importance": 1.0,
                    },
                    {
                        "line": "A[6,5:]",
                        "candidates": [],
                        "retained_lines": [],
                        "importance": 0.5,
                    },
                ],
            )

            bundle = jwd.build_family_bundle(artifacts_root=artifacts_root, family="A", board_size=19)

        decoded = self._decode_bundle(bundle)
        self.assertEqual(
            decoded["nodes"],
            [
                [2, True, False, False, False, 0],
                [1, True, True, True, True, 1000],
                [0, True, False, False, False, 0],
                [0, False, False, False, False, 0],
            ],
        )
        self.assertEqual(decoded["local_rows"], [[54, 1000], [44, 500], [34, 977]])
        self.assertEqual(decoded["local_child_bits"], [True, False, True])

    def test_write_joseki_bundles_writes_manifest_and_hashed_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            artifacts_root = repo_root / "artifacts" / "joseki"
            out_path = repo_root / "docs" / "data" / "joseki_current.json"
            stale_a = out_path.parent / "joseki_a.deadbeefcafe.bin"
            stale_o = out_path.parent / "joseki_o.deadbeefcafe.bin"
            stale_a.parent.mkdir(parents=True, exist_ok=True)
            stale_a.write_bytes(b"")
            stale_o.write_bytes(b"")

            self._write_artifact(
                artifacts_root=artifacts_root,
                family="A",
                nodes=[{"line": "", "candidates": [], "retained_lines": [], "importance": 1.0}],
            )
            self._write_artifact(
                artifacts_root=artifacts_root,
                family="O",
                nodes=[{"line": "", "candidates": [], "retained_lines": [], "importance": 1.0}],
            )

            jwd.write_joseki_bundles(artifacts_root=artifacts_root, out_path=out_path, board_size=19)

            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            bundle_a = self._decode_bundle((out_path.parent / str(manifest["bundles"]["A"])).read_bytes())
            bundle_o = self._decode_bundle((out_path.parent / str(manifest["bundles"]["O"])).read_bytes())

            self.assertFalse(stale_a.exists())
            self.assertFalse(stale_o.exists())

        self.assertEqual(manifest["version"], 1)
        self.assertRegex(str(manifest["bundles"]["A"]), r"^joseki_a\.[0-9a-f]{12}\.bin$")
        self.assertRegex(str(manifest["bundles"]["O"]), r"^joseki_o\.[0-9a-f]{12}\.bin$")
        self.assertEqual(bundle_a["nodes"], [[0, True, False, False, False, 0]])
        self.assertEqual(bundle_o["nodes"], [[0, True, False, False, False, 0]])


if __name__ == "__main__":
    unittest.main()

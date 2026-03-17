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
        node_control_offset = jwd.HEADER_STRUCT.size
        tenuki_drop_low_offset = node_control_offset + node_count
        tenuki_drop_high_offset = tenuki_drop_low_offset + node_count
        local_move_offset = tenuki_drop_high_offset + ((node_count * jwd.PACKED_TENUKI_DROP_HIGH_BITS + 7) // 8)
        local_child_bits = []
        for idx in range(node_count):
            word = payload[node_control_offset + idx]
            tenuki_drop = payload[tenuki_drop_low_offset + idx] + (
                self._read_bits(
                    payload,
                    offset=tenuki_drop_high_offset,
                    bit_offset=idx,
                    bits=1,
                ) << 9
            ) + (
                self._read_bits(
                    payload,
                    offset=tenuki_drop_high_offset,
                    bit_offset=node_count + idx,
                    bits=1,
                ) << 8
            )
            local_count = word & ((1 << jwd.PACKED_NODE_LOCAL_COUNT_BITS) - 1)
            local_children = bool((word >> jwd.PACKED_NODE_LOCAL_CHILDREN_SHIFT) & 0x1)
            local_child_bits.extend([local_children] * local_count)
            tenuki_present = idx > 0
            nodes.append(
                [
                    local_count,
                    bool((word >> jwd.PACKED_NODE_IS_CORE_SHIFT) & 0x1),
                    bool((word >> jwd.PACKED_NODE_TENUKI_RETAINED_SHIFT) & 0x1),
                    tenuki_present,
                    bool((word >> jwd.PACKED_NODE_TENUKI_CHILD_SHIFT) & 0x1),
                    1000 - tenuki_drop if tenuki_present else 0,
                    local_children,
                ]
            )
        first_local_drop_count = sum(1 for node in nodes if node[0] > 0)
        first_local_drop_offset = local_move_offset + local_row_count
        sibling_local_drop_offset = first_local_drop_offset + first_local_drop_count
        local_rows = []
        local_row_idx = 0
        first_idx = 0
        sibling_idx = 0
        for node in nodes:
            previous_drop = 0
            for local_idx in range(node[0]):
                if local_idx == 0:
                    previous_drop = payload[first_local_drop_offset + first_idx]
                    first_idx += 1
                else:
                    previous_drop = (
                        previous_drop + payload[sibling_local_drop_offset + sibling_idx]
                    ) & 0xFF
                    sibling_idx += 1
                local_rows.append([payload[local_move_offset + local_row_idx], 1000 - previous_drop])
                local_row_idx += 1
        self.assertEqual(first_idx, first_local_drop_count)
        self.assertEqual(sibling_idx, local_row_count - first_local_drop_count)
        offset = sibling_local_drop_offset + sibling_idx
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
                        ],
                        "retained_lines": ["A[6,5]"],
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
                        "candidates": [
                            {"kind": "tenuki", "stone_fraction": 0.8},
                            {"kind": "local", "local": [3, 4], "stone_fraction": 0.9},
                            {"kind": "local", "local": [2, 4], "stone_fraction": 0.95},
                        ],
                        "retained_lines": ["A[6,5:4,5:3,4]", "A[6,5:4,5:2,4]"],
                        "importance": 1.0,
                    },
                    {
                        "line": "A[6,5:]",
                        "candidates": [
                            {"kind": "tenuki", "stone_fraction": 0.7},
                        ],
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
                [1, True, False, False, False, 0, True],
                [1, True, True, True, True, 1000, True],
                [2, True, False, True, False, 800, False],
                [0, False, False, True, False, 700, False],
            ],
        )
        self.assertEqual(decoded["local_rows"], [[54, 1000], [34, 977], [23, 900], [13, 950]])
        self.assertEqual(decoded["local_child_bits"], [True, True, False, False])

    def test_write_joseki_bundles_maps_families(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts_root = root / "artifacts"
            out_path = root / "current.json"
            nodes = [{"line": "", "candidates": [], "retained_lines": [], "importance": 1.0}]
            self._write_artifact(artifacts_root=artifacts_root, family="A", nodes=nodes)
            self._write_artifact(artifacts_root=artifacts_root, family="O", nodes=nodes)
            jwd.write_joseki_bundles(artifacts_root=artifacts_root, out_path=out_path, board_size=19)
            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            written_family_codes = {
                family: self._decode_bundle((out_path.parent / name).read_bytes())["family_code"]
                for family, name in manifest["bundles"].items()
            }
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(written_family_codes, {"A": 1, "O": 2})
        self.assertTrue(manifest["bundles"]["A"].startswith("joseki_a."))
        self.assertTrue(manifest["bundles"]["O"].startswith("joseki_o."))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import joseki_website_data as jwd


class JosekiWebsiteDataTests(unittest.TestCase):
    def _decode_bundle(self, payload: bytes) -> dict[str, object]:
        magic, version, family_code, board_size, node_count, local_row_count = jwd.HEADER_STRUCT.unpack_from(payload, 0)
        self.assertEqual(magic, jwd.BUNDLE_MAGIC)
        nodes = []
        offset = jwd.HEADER_STRUCT.size
        for _ in range(node_count):
            parent_plus_1 = (
                payload[offset]
                | (payload[offset + 1] << 8)
                | (payload[offset + 2] << 16)
            )
            move_code = payload[offset + 3]
            flags = payload[offset + 4]
            tenuki_sf = struct.unpack_from("<H", payload, offset + 5)[0]
            nodes.append(
                [
                    parent_plus_1,
                    move_code,
                    flags,
                    tenuki_sf,
                ]
            )
            offset += jwd.NODE_ROW_SIZE
        local_rows = []
        for _ in range(local_row_count):
            move_code, stone_fraction = jwd.LOCAL_ROW_STRUCT.unpack_from(payload, offset)
            local_rows.append([move_code, stone_fraction])
            offset += jwd.LOCAL_ROW_STRUCT.size
        self.assertEqual(offset, len(payload))
        return {
            "version": version,
            "family_code": family_code,
            "board_size": board_size,
            "nodes": nodes,
            "local_rows": local_rows,
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
                            {"kind": "local", "local": [4, 5], "stone_fraction": 0.977},
                            {"kind": "tenuki", "stone_fraction": 1.0},
                        ],
                        "retained_lines": ["A[6,5:4,5]", "A[6,5:]"],
                        "importance": 0.99,
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
        self.assertEqual(decoded["version"], 1)
        self.assertEqual(decoded["family_code"], jwd.FAMILY_CODE_BY_NAME["A"])
        self.assertEqual(decoded["board_size"], 19)
        self.assertEqual(
            decoded["nodes"],
            [
                [0, 0, 0x21, 0],
                [1, 54, 0xE1, 1000],
                [2, 255, 0x00, 0],
            ],
        )
        self.assertEqual(decoded["local_rows"], [[54, 1000], [34, 977]])

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
        self.assertEqual(bundle_a["nodes"], [[0, 0, 0x20, 0]])
        self.assertEqual(bundle_o["nodes"], [[0, 0, 0x20, 0]])


if __name__ == "__main__":
    unittest.main()

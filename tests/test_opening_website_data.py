from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import opening_website_data as owd


class OpeningWebsiteDataTests(unittest.TestCase):
    def _decode_bundle(self, payload: bytes) -> dict[str, object]:
        magic, version, board_size, node_count, candidate_count = owd.HEADER_STRUCT.unpack_from(payload, 0)
        self.assertEqual(magic, owd.BUNDLE_MAGIC)
        nodes = []
        offset = owd.HEADER_STRUCT.size
        for _ in range(node_count):
            nodes.append(list(owd.NODE_STRUCT.unpack_from(payload, offset)))
            offset += owd.NODE_STRUCT.size
        candidates = []
        for _ in range(candidate_count):
            candidates.append(list(owd.CANDIDATE_STRUCT.unpack_from(payload, offset)))
            offset += owd.CANDIDATE_STRUCT.size
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
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "format": "move_tree",
                    "mode": "openings",
                    "board_size": board_size,
                    "root": 0,
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
                                    "retained": True,
                                    "prior": None,
                                    "tree_mover_winrate": None,
                                    "elo_loss": None,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_build_opening_bundle_compacts_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            artifacts_root = repo_root / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "format": "move_tree",
                        "mode": "openings",
                        "board_size": 11,
                        "root": 0,
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
                                        "retained": True,
                                        "prior": None,
                                        "raw_mover_winrate": 0.74,
                                        "tree_mover_winrate": 0.71,
                                        "elo_loss": 0.0,
                                    }
                                ],
                            },
                            {
                                "parent": 0,
                                "move": "a3",
                                "ply": 1,
                                "importance": 0.87654,
                                "candidates": [
                                    {
                                        "move": "c5",
                                        "child": None,
                                        "retained": False,
                                        "prior": 0.3333333,
                                        "raw_mover_winrate": 0.61234567,
                                        "tree_mover_winrate": 0.5234567,
                                        "elo_loss": 17.890123,
                                    },
                                    {
                                        "move": "d6",
                                        "child": 2,
                                        "retained": True,
                                        "prior": 0.4444444,
                                        "raw_mover_winrate": 0.71234567,
                                        "tree_mover_winrate": 0.6234567,
                                        "elo_loss": 7.890123,
                                    }
                                ],
                            },
                            {
                                "parent": 1,
                                "move": "d6",
                                "ply": 2,
                                "importance": 0.81234,
                                "candidates": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)
            decoded = self._decode_bundle(payload)

        self.assertEqual(decoded["version"], 1)
        self.assertEqual(decoded["board_size"], 11)
        self.assertEqual(
            decoded["nodes"],
            [
                [-1, -1, 1000, 1],
                [0, 22, 877, 1],
                [1, 58, 812, 0],
            ],
        )
        self.assertEqual(
            decoded["candidates"],
            [
                [22, 1, 65535, 710],
                [58, 1, 444, 623],
            ],
        )

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

    def test_build_opening_bundle_rejects_wrong_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "format": "move_tree",
                        "mode": "self_play",
                        "board_size": 11,
                        "root": 0,
                        "root_openings": ["a3"],
                        "completed": False,
                        "completed_ply": 1,
                        "nodes": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)

    def test_build_opening_bundle_rejects_wrong_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "openings"
            artifact_path = artifacts_root / "openings-s11.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "format": "move_tree",
                        "mode": "openings",
                        "board_size": 11,
                        "root": 0,
                        "root_openings": ["a3"],
                        "completed": False,
                        "completed_ply": 1,
                        "nodes": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported move-tree version"):
                owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)

if __name__ == "__main__":
    unittest.main()

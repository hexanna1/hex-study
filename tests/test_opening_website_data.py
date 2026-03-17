from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import opening_website_data as owd


class OpeningWebsiteDataTests(unittest.TestCase):
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
                    "board_size": board_size,
                    "root_openings": root_openings,
                    "completed": completed,
                    "completed_ply": completed_ply,
                    "nodes": [
                        {
                            "moves": [],
                            "importance": 1.0,
                            "candidates": [
                                {
                                    "move": root_openings[0],
                                    "prior": None,
                                    "mover_winrate": None,
                                    "elo_loss": None,
                                    "retained": True,
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
                        "board_size": 11,
                        "root_openings": ["a3", "a4"],
                        "completed": True,
                        "completed_ply": 3,
                        "nodes": [
                            {
                                "moves": [],
                                "importance": 1.0,
                                "candidates": [
                                    {
                                        "move": "a3",
                                        "prior": None,
                                        "mover_winrate": 0.74,
                                        "elo_loss": 0.0,
                                        "retained": True,
                                    }
                                ],
                            },
                            {
                                "moves": ["a3", "b4"],
                                "importance": 0.87654,
                                "candidates": [
                                    {
                                        "move": "c5",
                                        "prior": 0.3333333,
                                        "mover_winrate": 0.61234567,
                                        "elo_loss": 17.890123,
                                        "retained": False,
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = owd.build_opening_bundle(artifacts_root=artifacts_root, board_size=11)

        self.assertEqual(payload["board_size"], 11)
        self.assertEqual(payload["node_count"], 2)
        self.assertEqual(
            payload["nodes"],
            [
                {"m": "", "c": [["a3", None, 0.74, 1]], "i": 1.0},
                {"m": "a3b4", "c": [["c5", 0.333, 0.612, 0]], "i": 0.877},
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
            bundle_11 = json.loads(bundle_11_path.read_text(encoding="utf-8"))
            bundle_12 = json.loads(bundle_12_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["version"], 1)
        self.assertRegex(str(manifest["bundles"]["11"]), r"^opening_index\.[0-9a-f]{12}\.json$")
        self.assertRegex(str(manifest["bundles"]["12"]), r"^opening_index\.[0-9a-f]{12}\.json$")
        self.assertEqual(bundle_11["board_size"], 11)
        self.assertEqual(bundle_12["board_size"], 12)

if __name__ == "__main__":
    unittest.main()

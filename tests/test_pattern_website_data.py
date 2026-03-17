from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pattern_website_data as pwd


class PatternWebsiteDataTests(unittest.TestCase):
    def _write_catalog(self, batch_root: Path, patterns: list[dict]) -> None:
        batch_root.mkdir(parents=True, exist_ok=True)
        (batch_root / "catalog.json").write_text(
            json.dumps({"patterns": patterns}),
            encoding="utf-8",
        )

    def _write_tile(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_build_pattern_index_dedupes_identical_specs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            batch_root = repo_root / "artifacts" / "batch_a"
            tile_a = batch_root / "tiles" / "d03-k11.json"
            tile_b = batch_root / "tiles" / "d07-k11.json"
            payload = {
                "pattern": "+[0,0]-[]",
                "to_play": "blue",
                "cells": [
                    {"kind": "tenuki", "key": "tenuki", "stone_fraction": 1.0, "rank": 1},
                    {"kind": "local", "key": "-1,0", "stone_fraction": 0.75, "rank": 2, "local_rel": [-1, 0]},
                ],
            }
            other_payload = {
                "pattern": "+[0,0]-[]",
                "to_play": "blue",
                "cells": [
                    {"kind": "local", "key": "-2,0", "stone_fraction": 0.5, "rank": 1, "local_rel": [-2, 0]},
                ],
            }
            self._write_catalog(
                batch_root,
                [
                    {
                        "pattern": "+[0,0]-[]",
                        "study_delta": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                    {
                        "pattern": "+[]-[0,0]",
                        "study_delta": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k10k11",
                    },
                ],
            )
            self._write_tile(tile_a, payload)
            self._write_tile(tile_b, other_payload)

            out = pwd.build_pattern_index(artifacts_root=batch_root, repo_root=repo_root)

        self.assertEqual(out["pattern_count"], 1)
        entry = out["patterns"]["+[0,0]-[]"]
        self.assertEqual(entry["p"], "blue")
        self.assertEqual(entry["t"], 1000)
        self.assertEqual(
            entry["c"],
            [
                {"l": [-1, 0], "s": 750},
            ],
        )

    def test_build_pattern_index_rejects_conflicting_specs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            batch_root = repo_root / "artifacts" / "batch_a"
            tile_a = batch_root / "tiles" / "d03-k11.json"
            tile_b = batch_root / "tiles" / "d07-k11.json"
            payload_a = {
                "pattern": "+[0,0]-[]",
                "to_play": "blue",
                "cells": [
                    {"kind": "local", "key": "-1,0", "stone_fraction": 0.75, "rank": 1, "local_rel": [-1, 0]},
                ],
            }
            payload_b = {
                "pattern": "+[0,0]-[]",
                "to_play": "blue",
                "cells": [
                    {"kind": "local", "key": "-1,0", "stone_fraction": 0.5, "rank": 1, "local_rel": [-1, 0]},
                ],
            }
            self._write_catalog(
                batch_root,
                [
                    {
                        "pattern": "+[0,0]-[]",
                        "study_delta": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                    {
                        "pattern": "+[0,0]-[]",
                        "study_delta": 7,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                ],
            )
            self._write_tile(tile_a, payload_a)
            self._write_tile(tile_b, payload_b)

            with self.assertRaisesRegex(ValueError, "Conflicting tile specs"):
                pwd.build_pattern_index(artifacts_root=batch_root, repo_root=repo_root)

    def test_write_pattern_index_writes_manifest_and_hashed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            batch_root = repo_root / "artifacts" / "batch"
            tile = batch_root / "tiles" / "d03-k11.json"
            out_path = repo_root / "docs" / "data" / "current.json"
            self._write_catalog(
                batch_root,
                [
                    {
                        "pattern": "+[0,0]-[]",
                        "study_delta": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                ],
            )
            self._write_tile(
                tile,
                {
                    "pattern": "+[0,0]-[]",
                    "to_play": "blue",
                    "cells": [
                        {"kind": "local", "key": "-1,0", "stone_fraction": 0.75, "rank": 1, "local_rel": [-1, 0]},
                    ],
                },
            )
            stale_bundle = out_path.parent / "pattern_index.deadbeefcafe.json"
            stale_bundle.parent.mkdir(parents=True, exist_ok=True)
            stale_bundle.write_text("{}", encoding="utf-8")

            pwd.write_pattern_index(artifacts_root=batch_root, out_path=out_path)

            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            bundle_path = out_path.parent / str(manifest["bundle"])
            written = json.loads(bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["pattern_count"], 1)
        self.assertEqual(manifest["version"], 1)
        self.assertRegex(str(manifest["bundle"]), r"^pattern_index\.[0-9a-f]{12}\.json$")
        self.assertFalse(stale_bundle.exists())
        self.assertEqual(written["pattern_count"], 1)
        self.assertEqual(written["version"], 1)
        self.assertIn("+[0,0]-[]", written["patterns"])
        self.assertNotIn("tile_count", written)


if __name__ == "__main__":
    unittest.main()

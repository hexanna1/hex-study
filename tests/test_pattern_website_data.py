import json
from pathlib import Path
import tempfile
import unittest

import artifact_json as aj
import pattern_website_data as pwd


class PatternWebsiteDataTests(unittest.TestCase):
    def _write_catalog(self, root: Path, patterns: list[dict]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "catalog.json").write_text(
            json.dumps({"patterns": patterns}),
            encoding="utf-8",
        )

    def _write_tile(self, path: Path, payload: dict) -> None:
        aj.dump_pattern_tile(path, payload)

    def test_build_pattern_index_uses_existing_catalog_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            root = repo_root / "artifacts" / "batch"
            self._write_catalog(
                root,
                [
                    {
                        "pattern": "+[0,0]-[]",
                        "candidate_Δ_max": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                    {
                        "pattern": "+[]-[0,0]",
                        "candidate_Δ_max": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k10k11",
                    },
                ],
            )
            self._write_tile(
                root / "tiles" / "d03-k11.json",
                {
                    "pattern": "+[0,0]-[]",
                    "cells": [
                        {"kind": "tenuki", "stone_fraction": 1.0, "rank": 1},
                        {
                            "kind": "local",
                            "stone_fraction": 0.75,
                            "rank": 2,
                            "local_rel": [-1, 0],
                        },
                    ],
                },
            )
            out = pwd.build_pattern_index(artifacts_roots=[root], repo_root=repo_root)

        self.assertEqual(
            out,
            {
                "pattern_count": 1,
                "patterns": {
                    "+[0,0]-[]": {
                        "p": "red",
                        "t": 1000,
                        "c": [[-1, 0, 750]],
                    },
                },
            },
        )

    def test_build_pattern_index_rejects_equal_priority_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            root = repo_root / "artifacts" / "batch"
            rows = [
                {
                    "pattern": "+[0,0]-[]",
                    "candidate_Δ_max": delta,
                    "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                }
                for delta in (3, 7)
            ]
            self._write_catalog(root, rows)
            for delta, fraction in ((3, 0.75), (7, 0.5)):
                self._write_tile(
                    root / "tiles" / f"d{delta:02d}-k11.json",
                    {
                        "pattern": "+[0,0]-[]",
                        "cells": [
                            {
                                "kind": "local",
                                "stone_fraction": fraction,
                                "rank": 1,
                                "local_rel": [-1, 0],
                            },
                        ],
                    },
                )

            with self.assertRaisesRegex(ValueError, "Conflicting tile specs"):
                pwd.build_pattern_index(artifacts_roots=[root], repo_root=repo_root)

    def test_layered_index_partitions_by_move_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            base = repo_root / "artifacts" / "base"
            m6 = repo_root / "artifacts" / "m6"
            out_path = repo_root / "docs" / "data" / "current.json"
            base_rows = [
                {
                    "pattern": pattern,
                    "candidate_Δ_max": 3,
                    "hexworld_21": url,
                }
                for pattern, url in (
                    ("+[0,0]-[]", "https://hexworld.org/board/#21c1,k11"),
                    ("+[]-[0,0]", "https://hexworld.org/board/#21c1,k10k11"),
                )
            ]
            m6_pattern = "+[0,0:0,1:0,2]-[1,-1:1,0:1,1]"
            self._write_catalog(base, base_rows)
            self._write_catalog(
                m6,
                [
                    {
                        "pattern": m6_pattern,
                        "candidate_Δ_max": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,a1b1c1d1e1f1",
                    },
                ],
            )
            for path, pattern, local, fraction in (
                (base / "tiles" / "d03-k11.json", "+[0,0]-[]", [-1, 0], 0.75),
                (base / "tiles" / "d03-k10k11.json", "+[]-[0,0]", [-1, 0], 0.5),
                (m6 / "tiles" / "d03-a1b1c1d1e1f1.json", m6_pattern, [1, 0], 0.25),
            ):
                self._write_tile(
                    path,
                    {
                        "pattern": pattern,
                        "cells": [
                            {
                                "kind": "local",
                                "stone_fraction": fraction,
                                "rank": 1,
                                "local_rel": local,
                            },
                        ],
                    },
                )

            pwd.write_layered_pattern_index(
                artifacts_roots=[m6, base],
                out_path=out_path,
            )
            manifest = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["pattern_count"], 3)
        self.assertEqual(
            [
                (row["id"], row["min_moves"], row["max_moves"], row["pattern_count"])
                for row in manifest["layers"]
            ],
            [("base", 1, 5, 2), ("m6", 6, 6, 1)],
        )


if __name__ == "__main__":
    unittest.main()

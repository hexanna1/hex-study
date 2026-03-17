from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pattern_website_data as pwd


class PatternWebsiteDataTests(unittest.TestCase):
    def _read_bits(self, data: bytes, offset: int, bit_offset: int, bits: int) -> int:
        if bits == 0:
            return 0
        byte_offset = offset + (bit_offset // 8)
        shift = bit_offset % 8
        chunk = 0
        for idx in range(4):
            pos = byte_offset + idx
            if pos < len(data):
                chunk |= data[pos] << (8 * idx)
        return (chunk >> shift) & ((1 << bits) - 1)

    def _read_uvarint(self, data: bytes, offset: int) -> tuple[int, int]:
        value = 0
        shift = 0
        next_offset = int(offset)
        while next_offset < len(data):
            byte = data[next_offset]
            next_offset += 1
            value += (byte & 0x7F) << shift
            if byte < 0x80:
                return value, next_offset
            shift += 7
        raise AssertionError("unterminated varint")

    def _read_bitplane_value(self, data: bytes, offset: int, value_count: int, value_idx: int, bits: int) -> int:
        value = 0
        for bit in range(bits):
            value = (value << 1) | self._read_bits(data, offset, (bit * value_count) + value_idx, 1)
        return value

    def _decode_presence_row(self, data: bytes, offset: int, pattern_count: int) -> tuple[list[int], int]:
        mode = data[offset]
        next_offset = offset + 1
        if mode == pwd.PRESENCE_MODE_ALL:
            return [1] * pattern_count, next_offset
        if mode == pwd.PRESENCE_MODE_BITMAP:
            present = [
                self._read_bits(data, next_offset, idx, 1)
                for idx in range(pattern_count)
            ]
            next_offset += ((pattern_count + 7) // 8)
            return present, next_offset
        if mode in {pwd.PRESENCE_MODE_PRESENT_GAPS, pwd.PRESENCE_MODE_ABSENT_GAPS}:
            present = [0] * pattern_count if mode == pwd.PRESENCE_MODE_PRESENT_GAPS else [1] * pattern_count
            count, next_offset = self._read_uvarint(data, next_offset)
            idx = -1
            for _ in range(count):
                gap, next_offset = self._read_uvarint(data, next_offset)
                idx += gap + 1
                present[idx] = 1 if mode == pwd.PRESENCE_MODE_PRESENT_GAPS else 0
            return present, next_offset
        raise AssertionError(f"bad presence mode: {mode}")

    def _decode_bundle(self, payload: bytes) -> dict[str, object]:
        magic, version, fraction_mode, pattern_count, cell_count, key_stream_size = pwd.HEADER_STRUCT.unpack_from(payload, 0)
        self.assertEqual(magic, pwd.BUNDLE_MAGIC)
        self.assertEqual(fraction_mode, pwd.FRACTION_MODE_COORD_MAJOR_BITPLANE)
        offset = pwd.HEADER_STRUCT.size
        tenuki_offset = offset
        offset += ((pattern_count * pwd.PACKED_STONE_FRACTION_BITS) + 7) // 8
        tenukis = [
            self._read_bitplane_value(payload, tenuki_offset, pattern_count, idx, pwd.PACKED_STONE_FRACTION_BITS)
            for idx in range(pattern_count)
        ]
        pair_count = pwd.PAIR_COUNT_STRUCT.unpack_from(payload, offset)[0]
        offset += pwd.PAIR_COUNT_STRUCT.size
        pair_rows = []
        for _ in range(pair_count):
            q, r = pwd.PAIR_ROW_STRUCT.unpack_from(payload, offset)
            pair_rows.append([q, r])
            offset += pwd.PAIR_ROW_STRUCT.size
        presence_size = int.from_bytes(payload[offset:offset + 4], "little")
        offset += 4
        presence_offset = offset
        offset += presence_size
        coord_rows = []
        presence_cursor = presence_offset
        fraction_start = 0
        for pair in pair_rows:
            present, presence_cursor = self._decode_presence_row(payload, presence_cursor, pattern_count)
            coord_rows.append((pair, present, fraction_start))
            fraction_start += sum(present)
        self.assertEqual(presence_cursor, presence_offset + presence_size)
        self.assertEqual(fraction_start, cell_count)
        fraction_offset = offset
        offset += ((cell_count * pwd.PACKED_STONE_FRACTION_BITS) + 7) // 8
        cells_by_entry = [[] for _ in range(pattern_count)]
        for pair, present, row_fraction_start in coord_rows:
            seen_before = 0
            for entry_idx, is_present in enumerate(present):
                if not is_present:
                    continue
                fraction_idx = row_fraction_start + seen_before
                stone_fraction = self._read_bitplane_value(
                    payload,
                    fraction_offset,
                    cell_count,
                    fraction_idx,
                    pwd.PACKED_STONE_FRACTION_BITS,
                )
                cells_by_entry[entry_idx].append([*pair, stone_fraction])
                seen_before += 1
        key_stream = payload[offset:]
        self.assertEqual(len(key_stream), key_stream_size)
        return {
            "version": version,
            "pattern_count": pattern_count,
            "tenukis": tenukis,
            "pair_rows": pair_rows,
            "cells_by_entry": cells_by_entry,
            "key_stream": key_stream,
        }

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
                [-1, 0, 750],
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
                        "candidate_Δ_max": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                    {
                        "pattern": "+[0,0]-[]",
                        "candidate_Δ_max": 7,
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
                        "candidate_Δ_max": 3,
                        "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                    },
                ],
            )
            self._write_tile(
                tile,
                {
                    "pattern": "+[0,0]-[]",
                    "to_play": "red",
                    "cells": [
                        {"kind": "tenuki", "key": "tenuki", "stone_fraction": 1.0, "rank": 0},
                        {"kind": "local", "key": "-1,0", "stone_fraction": 0.75, "rank": 1, "local_rel": [-1, 0]},
                        {"kind": "local", "key": "0,-1", "stone_fraction": 0.625, "rank": 2, "local_rel": [0, -1]},
                    ],
                },
            )
            stale_bundle = out_path.parent / "pattern_index.deadbeefcafe.bin"
            stale_bundle.parent.mkdir(parents=True, exist_ok=True)
            stale_bundle.write_bytes(b"")

            pwd.write_pattern_index(artifacts_root=batch_root, out_path=out_path)

            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            bundle_path = out_path.parent / str(manifest["bundle"])
            written = self._decode_bundle(bundle_path.read_bytes())

            self.assertFalse(stale_bundle.exists())

        self.assertEqual(manifest["pattern_count"], 1)
        self.assertEqual(manifest["version"], 1)
        self.assertRegex(str(manifest["bundle"]), r"^pattern_index\.[0-9a-f]{12}\.bin$")
        self.assertEqual(written["version"], 1)
        self.assertEqual(written["pattern_count"], 1)
        self.assertEqual(written["tenukis"], [1000])
        self.assertEqual(written["pair_rows"], [[-1, 0], [0, -1]])
        self.assertEqual(written["cells_by_entry"], [[[-1, 0, 750], [0, -1, 625]]])
        self.assertEqual(written["key_stream"], bytes([0, 2, 1, 0x88]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pattern_website_data as pwd
try:
    from bundle_test_utils import read_little_endian_bits
except ModuleNotFoundError:  # pragma: no cover - package-style unittest invocation
    from tests.bundle_test_utils import read_little_endian_bits


class PatternWebsiteDataTests(unittest.TestCase):
    def _read_bits(self, data: bytes, offset: int, bit_offset: int, bits: int) -> int:
        return read_little_endian_bits(data, offset=offset, bit_offset=bit_offset, bits=bits, chunk_bytes=4)

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

    def _read_signed_bitplane_value(self, data: bytes, offset: int, value_count: int, value_idx: int, bits: int) -> int:
        value = self._read_bitplane_value(data, offset, value_count, value_idx, bits)
        return value // 2 if value % 2 == 0 else -((value + 1) // 2)

    def _decode_presence_rows(
        self,
        data: bytes,
        pattern_count: int,
        pair_rows: list[list[int]],
    ) -> list[list[int]]:
        pair_count = len(pair_rows)
        flag_bytes = (pattern_count + 7) // 8
        bitmap_bytes = (pair_count + 7) // 8
        offset = flag_bytes
        present_rows = [[0] * pattern_count for _ in range(pair_count)]
        previous = [0] * pair_count
        for entry_idx in range(pattern_count):
            current = list(previous)
            use_bitmap = self._read_bits(data, 0, entry_idx, 1) == 1
            if use_bitmap:
                self.assertLessEqual(offset + bitmap_bytes, len(data))
                for pair_idx in range(pair_count):
                    if self._read_bits(data, offset, pair_idx, 1):
                        current[pair_idx] = 1 - current[pair_idx]
                offset += bitmap_bytes
            else:
                count, offset = self._read_uvarint(data, offset)
                pair_idx = -1
                for _ in range(count):
                    gap, offset = self._read_uvarint(data, offset)
                    pair_idx += gap + 1
                    self.assertGreaterEqual(pair_idx, 0)
                    self.assertLess(pair_idx, pair_count)
                    current[pair_idx] = 1 - current[pair_idx]
            for pair_idx, is_present in enumerate(current):
                previous[pair_idx] = is_present
                present_rows[pair_idx][entry_idx] = is_present
        self.assertEqual(offset, len(data))
        return present_rows

    def _decode_bundle(self, payload: bytes) -> dict[str, object]:
        magic, version, fraction_mode, pattern_count, cell_count, key_stream_size = pwd.HEADER_STRUCT.unpack_from(payload, 0)
        self.assertEqual(magic, pwd.BUNDLE_MAGIC)
        self.assertEqual(fraction_mode, pwd.FRACTION_MODE_ROW_COORD_BITPLANE)
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
        decoded_presence = self._decode_presence_rows(
            payload[presence_offset:presence_offset + presence_size],
            pattern_count,
            pair_rows,
        )
        fraction_start = sum(sum(present) for present in decoded_presence)
        self.assertEqual(fraction_start, cell_count)
        fraction_offset = offset
        key_stream_offset = len(payload) - key_stream_size
        key_stream = payload[key_stream_offset:]
        self.assertEqual(len(key_stream), key_stream_size)
        row_mean_offset = fraction_offset
        offset += ((pattern_count * pwd.PACKED_STONE_FRACTION_BITS) + 7) // 8
        pair_mean_offset = offset
        offset += ((pair_count * pwd.PACKED_STONE_FRACTION_BITS) + 7) // 8
        global_mean = int.from_bytes(payload[offset:offset + 2], "little")
        offset += 2
        residual_bits = payload[offset]
        offset += 1
        residual_offset = offset
        offset += ((cell_count * residual_bits) + 7) // 8
        self.assertEqual(offset, key_stream_offset)
        cells_by_entry = [[] for _ in range(pattern_count)]
        row_fraction_start = 0
        for pair_idx, (pair, present) in enumerate(zip(pair_rows, decoded_presence)):
            seen_before = 0
            for entry_idx, is_present in enumerate(present):
                if not is_present:
                    continue
                fraction_idx = row_fraction_start + seen_before
                row_mean = self._read_bitplane_value(
                    payload,
                    row_mean_offset,
                    pattern_count,
                    entry_idx,
                    pwd.PACKED_STONE_FRACTION_BITS,
                )
                pair_mean = self._read_bitplane_value(
                    payload,
                    pair_mean_offset,
                    pair_count,
                    pair_idx,
                    pwd.PACKED_STONE_FRACTION_BITS,
                )
                residual = self._read_signed_bitplane_value(
                    payload,
                    residual_offset,
                    cell_count,
                    fraction_idx,
                    residual_bits,
                )
                stone_fraction = row_mean + pair_mean - global_mean + residual
                cells_by_entry[entry_idx].append([*pair, stone_fraction])
                seen_before += 1
            row_fraction_start += seen_before
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

            out = pwd.build_pattern_index(artifacts_roots=[batch_root], repo_root=repo_root)

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
                pwd.build_pattern_index(artifacts_roots=[batch_root], repo_root=repo_root)

    def test_write_layered_pattern_index_partitions_combined_roots_by_move_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            base_root = repo_root / "artifacts" / "base"
            ext_root = repo_root / "artifacts" / "ext"
            out_path = repo_root / "docs" / "data" / "current.json"
            row_a = {"pattern": "+[0,0]-[]", "candidate_Δ_max": 3, "hexworld_21": "https://hexworld.org/board/#21c1,k11"}
            row_b = {
                "pattern": "+[0,0:0,1:0,2]-[1,-1:1,0:1,1]",
                "candidate_Δ_max": 3,
                "hexworld_21": "https://hexworld.org/board/#21c1,a1b1c1d1e1f1",
            }
            self._write_catalog(base_root, [row_a])
            self._write_catalog(ext_root, [row_b])
            self._write_tile(
                base_root / "tiles" / "d03-k11.json",
                {"pattern": "+[0,0]-[]", "to_play": "red", "cells": [
                    {"kind": "local", "key": "-1,0", "stone_fraction": 0.75, "rank": 1, "local_rel": [-1, 0]},
                ]},
            )
            self._write_tile(
                ext_root / "tiles" / "d03-a1b1c1d1e1f1.json",
                {"pattern": "+[0,0:0,1:0,2]-[1,-1:1,0:1,1]", "to_play": "red", "cells": [
                    {"kind": "local", "key": "1,0", "stone_fraction": 0.25, "rank": 1, "local_rel": [1, 0]},
                ]},
            )
            stale_bundle = out_path.parent / "pattern_index.000000000000.bin"
            stale_bundle.parent.mkdir(parents=True, exist_ok=True)
            stale_bundle.write_bytes(b"")

            pwd.write_layered_pattern_index(
                artifacts_roots=[ext_root, base_root],
                out_path=out_path,
            )

            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            layers = manifest["layers"]
            bundle_paths = [out_path.parent / str(row["bundle"]) for row in layers]
            base_written = self._decode_bundle(bundle_paths[0].read_bytes())
            m6_written = self._decode_bundle(bundle_paths[1].read_bytes())

            self.assertFalse(stale_bundle.exists())
            self.assertTrue(all(path.exists() for path in bundle_paths))

        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["pattern_count"], 2)
        self.assertEqual([row["id"] for row in layers], ["base", "m6"])
        self.assertEqual([row["pattern_count"] for row in layers], [1, 1])
        self.assertEqual([(row["min_moves"], row["max_moves"]) for row in layers], [(1, 5), (6, 6)])
        self.assertTrue(all(str(row["bundle"]).startswith("pattern_index.") for row in layers))
        self.assertEqual(base_written["pattern_count"], 1)
        self.assertEqual(base_written["cells_by_entry"], [[[-1, 0, 750]]])
        self.assertEqual(m6_written["pattern_count"], 1)
        self.assertEqual(m6_written["cells_by_entry"], [[[1, 0, 250]]])


if __name__ == "__main__":
    unittest.main()

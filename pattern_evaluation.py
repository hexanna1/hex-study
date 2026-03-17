"""Pattern-study evaluation with shared baselines and transient pattern samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Iterator

import hex_board as hb
import pattern_scoring as scoring
import raw_nn as nn


RAW_NN_BATCH_CHUNK_SIZE = 5000


@dataclass(slots=True)
class PatternEvaluator:
    board_size: int
    to_play: str
    cache_path: Path
    cache: dict[str, Any] = field(init=False)
    candidate_values: dict[str, float] = field(default_factory=dict, init=False)
    dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.cache = nn.load_cache(self.cache_path)

    def checkpoint(self) -> None:
        if self.dirty:
            nn.save_cache(self.cache_path, self.cache, merge=True)
            self.dirty = False

    def roots(
        self, *, position_inputs: list[str], persistent_keys: set[str],
    ) -> tuple[dict[str, dict[str, Any]], int, float]:
        started = monotonic()
        payloads = {}
        missing = []
        hits = 0
        for position in dict.fromkeys(position_inputs):
            key = hb.position_cache_key(position)
            cached = self.cache.get(key) if key in persistent_keys else None
            if isinstance(cached, dict):
                payloads[position] = nn.decode_payload(cached)
                hits += 1
            else:
                missing.append(position)
        if missing:
            fetched = nn.evaluate(
                position_inputs=missing, board_size=self.board_size,
                move_limit=self.board_size**2 + 1,
            )
            for position in missing:
                payload = fetched[position]
                key = hb.position_cache_key(position)
                if key in persistent_keys:
                    self.cache[key] = nn.encode_payload(payload)
                    self.dirty = True
                payloads[position] = payload
        return payloads, hits, monotonic() - started

    def _children(self, position: str, moves: list[str]) -> Iterator[tuple[str, str, str]]:
        size, red, blue, to_play = hb.position_state(position)
        for move in moves:
            child = hb.position_after_move_from_state(
                size=size, red=red, blue=blue, to_play=to_play, move=move,
            )
            yield move, child, hb.serialized_position_key(child)

    def fill_candidates(
        self, requests: list[tuple[str, list[str]]], *, persistent_parent_positions: set[str],
    ) -> int:
        missing = {}
        persistent_keys = set()
        for position, moves in requests:
            persist = hb.position_cache_key(position) in persistent_parent_positions
            for _move, child, key in self._children(position, moves):
                if persist:
                    persistent_keys.add(key)
                cached = self.cache.get(key) if persist else None
                if isinstance(cached, dict):
                    self.candidate_values[key] = nn.decode_millionths(cached["r"])
                elif cached is not None:
                    self.candidate_values[key] = nn.decode_millionths(cached)
                elif key not in self.candidate_values:
                    missing.setdefault(key, child)
        items = list(missing.items())
        for start in range(0, len(items), RAW_NN_BATCH_CHUNK_SIZE):
            chunk = items[start : start + RAW_NN_BATCH_CHUNK_SIZE]
            fetched = nn.evaluate(
                position_inputs=[child for _key, child in chunk],
                board_size=self.board_size, move_limit=1,
            )
            for key, child in chunk:
                winrate = fetched[child]["r"]
                self.candidate_values[key] = winrate
                if key in persistent_keys:
                    self.cache[key] = nn.encode_millionths(winrate)
                    self.dirty = True
        return len(missing)

    def candidate_rows(
        self, *, position_input: str, moves: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = []
        for index, (move, _child, key) in enumerate(self._children(position_input, moves)):
            red_winrate = self.candidate_values[key]
            winrate = red_winrate if self.to_play == "red" else 1.0 - red_winrate
            rows.append({
                "move": move, "n": 1, "mean_winrate": winrate,
                "stdev_winrate": 0.0, "min_winrate": winrate, "max_winrate": winrate,
                "_idx": index,
            })
        scoring.attach_stone_fractions(
            rows, position_input=position_input, allow_first_row_proxy_fallback=True,
        )
        rows.sort(key=lambda row: (-row["mean_winrate"], row["_idx"], row["move"]))
        return rows, {"source": "raw_nn_cache_batchrawnn", "hexworld": position_input, "rows": rows}

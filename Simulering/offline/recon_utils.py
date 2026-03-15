"""Utilities for finding and loading recon files."""
from __future__ import annotations

import json
from pathlib import Path

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# Fingerprint -> bot count mapping (known maps)
_DIFFICULTY_BOT_COUNT = {"easy": 1, "medium": 3, "hard": 5, "expert": 10, "nightmare": 20}


def find_latest_recon(difficulty: str) -> Path | None:
    """Find the most recent recon file for a difficulty level.

    Matches by bot_count in the recon JSON (1=easy, 3+=medium).
    Returns None if no matching file is found.
    """
    target_bots = _DIFFICULTY_BOT_COUNT.get(difficulty)
    if target_bots is None:
        return None

    candidates: list[Path] = []
    for p in _LOGS_DIR.glob("*_recon.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            bot_count = data.get("bot_count", len(data.get("spawn_positions", [])))
            target = _DIFFICULTY_BOT_COUNT.get(difficulty, 0)
            if bot_count == target:
                candidates.append(p)
        except (json.JSONDecodeError, OSError):
            continue

    if not candidates:
        return None

    # Prefer recon with MOST orders (best data), then newest file as tiebreak
    def _score(p: Path) -> tuple[int, float]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            n_orders = len(data.get("order_sequence", []))
        except (json.JSONDecodeError, OSError):
            n_orders = 0
        return (n_orders, p.stat().st_mtime)

    return max(candidates, key=_score)

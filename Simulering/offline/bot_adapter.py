"""
BotAdapter: runs the live bot's full pipeline inside the offline Simulator.

Bridges the Simulator (expects a strategy callable) with the live bot's
Coordinator (processes game state dicts). This lets us test the actual
competition bot offline — same code path as a live game, zero tokens.

Usage:
    from offline.simulator import Simulator
    from offline.bot_adapter import BotAdapter

    sim = Simulator.from_recon_data(recon_data)

    adapter = BotAdapter()
    result = sim.run(adapter)
    recon = adapter.finalize(result)   # returns recon dict, optionally saves

    # Compare with simple strategy:
    from offline.strategy import ParameterizedStrategy, StrategyParams
    simple = ParameterizedStrategy(StrategyParams(), sim.width, ...)
    baseline = sim.run(simple)
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path so `bot` package resolves
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bot.coordinator import Coordinator

logger = logging.getLogger(__name__)


class BotAdapter:
    """
    Wraps the live bot's Coordinator as a Simulator-compatible strategy.

    Implements __call__(state_dict) -> response_dict, which is the interface
    Simulator.run() expects for dict-based strategies.

    The adapter manages Coordinator lifecycle:
    - Fresh Coordinator per game (call reset() between games)
    - Optional recon data capture via finalize()
    - Configurable log suppression for batch runs
    """

    def __init__(
        self,
        *,
        save_recon: bool = False,
        logs_dir: Optional[Path] = None,
        suppress_logs: bool = False,
        force_plan: Optional[dict] = None,
        config=None,
    ) -> None:
        """
        Args:
            save_recon: If True, finalize() writes recon/plan files to logs_dir.
            logs_dir: Directory for recon output. Defaults to temp dir when
                      save_recon=False to prevent accidental plan file pickup.
            suppress_logs: If True, set bot loggers to WARNING during run.
            force_plan: If set, Coordinator uses this plan (replay mode)
                        instead of detecting from disk. Used by the optimizer.
            config: CoordinatorConfig for tunable parameters. Used by optimizer.
        """
        self._save_recon = save_recon
        self._suppress_logs = suppress_logs
        self._force_plan = force_plan
        self._config = config

        if logs_dir is not None:
            self._logs_dir = Path(logs_dir)
        elif save_recon:
            self._logs_dir = _PROJECT_ROOT / "logs"
        else:
            self._logs_dir = Path(tempfile.mkdtemp(prefix="simbot_"))

        self._coordinator: Optional[Coordinator] = None
        self._original_log_levels: dict[str, int] = {}

    def __call__(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Strategy interface for Simulator.run()."""
        if self._coordinator is None:
            self._coordinator = self._make_coordinator()
            if self._suppress_logs:
                self._quiet_bot_loggers()

        return self._coordinator.on_game_state(state_dict)

    def finalize(self, result: dict[str, Any]) -> Optional[dict]:
        """
        Call after sim.run() completes. Triggers recon data capture.

        Returns the raw recon data dict (or None if coordinator wasn't used).
        When save_recon=True, also writes recon/plan files to disk via Coordinator.
        """
        if self._coordinator is None:
            return None

        total = result.get("rounds_used", 300)
        score = result.get("score", 0)

        # Get recon data before finalize_game (which also calls finalize internally)
        recon = self._coordinator._game_logger.finalize(total, score)

        if self._save_recon:
            self._coordinator.finalize_game(total_rounds=total, final_score=score)

        if self._suppress_logs:
            self._restore_bot_loggers()

        return recon

    def reset(self) -> None:
        """Reset for a new game. Call between consecutive sim.run() calls."""
        if self._coordinator is not None:
            self._coordinator.reset()
        self._coordinator = None

        if self._suppress_logs:
            self._restore_bot_loggers()

    def _make_coordinator(self) -> Coordinator:
        coord = Coordinator()
        coord._logs_dir = self._logs_dir
        return coord

    def _quiet_bot_loggers(self) -> None:
        """Suppress bot package logging for batch simulation runs."""
        for name in ("bot", "bot.coordinator", "bot.engine", "bot.strategy",
                     "bot.recon"):
            log = logging.getLogger(name)
            self._original_log_levels[name] = log.level
            log.setLevel(logging.WARNING)

    def _restore_bot_loggers(self) -> None:
        for name, level in self._original_log_levels.items():
            logging.getLogger(name).setLevel(level)
        self._original_log_levels.clear()

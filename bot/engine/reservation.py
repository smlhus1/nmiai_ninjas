"""
ReservationTable: tracks space-time reservations for multi-bot path planning.

Each cell (x, y, t) can be reserved by at most one bot. When a bot's path ends,
its final position stays reserved for all future timesteps up to the horizon.
"""

from __future__ import annotations

from bot.models import Pos


class ReservationTable:
    """Space-time reservation table for collision-free path planning."""

    def __init__(self, horizon: int = 50) -> None:
        self._table: dict[tuple[int, int, int], int] = {}  # (x, y, t) -> bot_id
        self._bot_paths: dict[int, list[Pos]] = {}  # bot_id -> [pos_at_t0, pos_at_t1, ...]
        self._start_times: dict[int, int] = {}  # bot_id -> start timestep of path
        self._horizon = horizon

    def is_free(self, pos: Pos, t: int, bot_id: int) -> bool:
        """Check if (pos, t) is free for bot_id (own reservations don't block)."""
        occupant = self._table.get((pos[0], pos[1], t))
        return occupant is None or occupant == bot_id

    def has_following_conflict(self, pos: Pos, t: int, bot_id: int) -> bool:
        """Check for sequential-execution following conflict.

        In ID-order sequential execution, lower-ID bots are processed first.
        A lower-ID bot cannot move to a cell occupied by a higher-ID bot at time t,
        even if the higher-ID bot moves away at t+1 (it hasn't been processed yet).

        Returns True if a higher-ID bot occupies pos at time t.
        """
        occupant = self._table.get((pos[0], pos[1], t))
        return occupant is not None and occupant != bot_id and occupant > bot_id

    def has_edge_conflict(self, pos_from: Pos, pos_to: Pos, t: int, bot_id: int) -> bool:
        """Check for swap/edge conflict: another bot moving pos_to->pos_from at time t->t+1."""
        # At time t, another bot is at pos_to and at time t+1 it moves to pos_from
        occupant_at_to = self._table.get((pos_to[0], pos_to[1], t))
        if occupant_at_to is None or occupant_at_to == bot_id:
            return False
        # Check if that bot moves to pos_from at t+1
        other_path = self._bot_paths.get(occupant_at_to)
        if other_path is None:
            return False
        other_start = self._start_times.get(occupant_at_to, 0)
        other_idx = t + 1 - other_start
        if 0 <= other_idx < len(other_path):
            return other_path[other_idx] == pos_from
        elif other_path:
            # Past end of path: bot stays at last position
            return other_path[-1] == pos_from
        return False

    def reserve_path(self, bot_id: int, path: list[Pos], start_t: int,
                     max_idle: int | None = None,
                     safe_mode: bool = False) -> None:
        """Reserve all cells along a path. Clears previous reservations for this bot.

        Args:
            max_idle: If set, limit how many extra rounds to reserve the final
                      position beyond path end. None = extend to full horizon
                      (original behavior). Use a small value for MAPF planning
                      to prevent idle bots from permanently blocking corridors.
            safe_mode: If True, never overwrite cells reserved by other bots.
                       Use for MAPF planning to prevent cascading reservation
                       overwrites. Gaps in extension are left where other bots
                       pass through.
        """
        self.clear_bot(bot_id)
        self._bot_paths[bot_id] = list(path)
        self._start_times[bot_id] = start_t

        for i, pos in enumerate(path):
            t = start_t + i
            if t <= self._horizon:
                self._table[(pos[0], pos[1], t)] = bot_id

        # Reserve final position for remaining timesteps (bot parks there)
        if path:
            final = path[-1]
            end_t = start_t + len(path) - 1
            extend_to = self._horizon if max_idle is None else min(end_t + max_idle, self._horizon)
            for t in range(end_t + 1, extend_to + 1):
                if safe_mode:
                    # Don't overwrite other bots' path entries during extension
                    existing = self._table.get((final[0], final[1], t))
                    if existing is not None and existing != bot_id:
                        continue
                self._table[(final[0], final[1], t)] = bot_id

    def clear_bot(self, bot_id: int) -> None:
        """Remove all reservations for a bot."""
        if bot_id not in self._bot_paths:
            return
        # Remove from table
        keys_to_remove = [k for k, v in self._table.items() if v == bot_id]
        for k in keys_to_remove:
            del self._table[k]
        del self._bot_paths[bot_id]
        self._start_times.pop(bot_id, None)

    def get_planned_pos(self, bot_id: int, t: int) -> Pos | None:
        """Get planned position for a bot at timestep t."""
        path = self._bot_paths.get(bot_id)
        if path is None:
            return None
        start = self._start_times.get(bot_id, 0)
        idx = t - start
        if idx < 0:
            return None
        if idx < len(path):
            return path[idx]
        # Past end of path: stays at final position
        return path[-1] if path else None

    def append_path(self, bot_id: int, path: list[Pos], start_t: int,
                    max_idle: int | None = None,
                    safe_mode: bool = False) -> None:
        """Reserve cells along a path WITHOUT clearing previous reservations.

        Use this when appending a new trip segment to an existing bot path.
        Does NOT call clear_bot — preserves all previous reservations.
        Updates _bot_paths and _start_times to the combined view.
        """
        import logging
        _logger = logging.getLogger(__name__)

        # Extend the bot's known path (for edge conflict detection)
        existing = self._bot_paths.get(bot_id, [])
        existing_start = self._start_times.get(bot_id, start_t)
        # Compute combined path
        if existing:
            # Fill gap if any
            existing_end_t = existing_start + len(existing)
            if start_t > existing_end_t:
                gap = start_t - existing_end_t
                existing.extend([existing[-1]] * gap)
            self._bot_paths[bot_id] = existing + list(path)
        else:
            self._bot_paths[bot_id] = list(path)
            self._start_times[bot_id] = start_t

        for i, pos in enumerate(path):
            t = start_t + i
            if t <= self._horizon:
                existing_owner = self._table.get((pos[0], pos[1], t))
                if existing_owner is not None and existing_owner != bot_id:
                    _logger.error("OVERWRITE: bot %d at (%d,%d,t=%d) overwrites bot %d",
                                  bot_id, pos[0], pos[1], t, existing_owner)
                self._table[(pos[0], pos[1], t)] = bot_id

        # Extension for final position
        if path:
            final = path[-1]
            end_t = start_t + len(path) - 1
            extend_to = self._horizon if max_idle is None else min(end_t + max_idle, self._horizon)
            for t in range(end_t + 1, extend_to + 1):
                if safe_mode:
                    existing_owner = self._table.get((final[0], final[1], t))
                    if existing_owner is not None and existing_owner != bot_id:
                        continue
                self._table[(final[0], final[1], t)] = bot_id

    def reserve_position(self, bot_id: int, pos: Pos, t: int) -> None:
        """Reserve a single (pos, t) cell. Does NOT set up path or final-pos extension."""
        self._table[(pos[0], pos[1], t)] = bot_id

    def clear_all(self) -> None:
        """Clear all reservations."""
        self._table.clear()
        self._bot_paths.clear()
        self._start_times.clear()

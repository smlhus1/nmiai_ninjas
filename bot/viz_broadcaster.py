"""
WebSocket broadcaster for game visualization.

Runs a WebSocket server in a background thread. Coordinator calls
send_state() each round to push game state to connected viz clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

from bot.models import GameState, Pos
from bot.strategy.task import BotAssignment, TaskType

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8765


class VizBroadcaster:
    """Broadcasts game state to visualization clients via WebSocket."""

    def __init__(self, port: int = _DEFAULT_PORT) -> None:
        self._port = port
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_state: Optional[dict] = None
        self._all_states: list[dict] = []  # For replay mode (sim)
        # Stored once on first round
        self._shelf_positions: list[list[int]] = []
        self._one_way: dict[str, list[int]] = {}

    def start(self) -> None:
        """Start WebSocket server in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info("Viz broadcaster started on ws://localhost:%d", self._port)

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def set_metadata(
        self,
        shelf_positions: frozenset[Pos] | None = None,
        one_way: dict[Pos, tuple[int, int]] | None = None,
    ) -> None:
        """Set static metadata (shelves, one-way rules) once."""
        if shelf_positions:
            self._shelf_positions = [list(p) for p in sorted(shelf_positions)]
        if one_way:
            self._one_way = {
                f"{p[0]},{p[1]}": [d[0], d[1]] for p, d in one_way.items()
            }

    def send_state(
        self,
        state: GameState,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Serialize and record current game state."""
        data = self._serialize(state, assignments)
        self._last_state = data
        self._all_states.append(data)

        # Broadcast to connected WS clients (live mode)
        if self._clients and self._loop:
            msg = json.dumps(data)
            asyncio.run_coroutine_threadsafe(
                self._broadcast(msg), self._loop
            )

    def _serialize(
        self,
        state: GameState,
        assignments: dict[int, BotAssignment],
    ) -> dict:
        """Convert game state to JSON-serializable dict."""
        bots = []
        for bot in sorted(state.bots, key=lambda b: b.id):
            a = assignments.get(bot.id)
            task_type = None
            target = None
            item_type = None
            nav_override = None
            if a and a.task:
                task_type = a.task.task_type.value if a.task.task_type else None
                target = list(a.effective_target) if a.effective_target else None
                item_type = a.task.item_type
            if a and a.navigation_override:
                nav_override = list(a.navigation_override)

            bots.append({
                "id": bot.id,
                "position": list(bot.position),
                "inventory": list(bot.inventory),
                "task_type": task_type,
                "target": target,
                "item_type": item_type,
                "nav_override": nav_override,
            })

        orders = []
        for order in state.orders:
            orders.append({
                "id": order.id,
                "status": order.status.value,
                "items_required": list(order.items_required),
                "items_delivered": list(order.items_delivered),
                "items_remaining": list(order.items_remaining),
            })

        items = []
        for item in state.items:
            items.append({
                "id": item.id,
                "type": item.type,
                "position": list(item.position),
            })

        return {
            "type": "game_state",
            "round": state.round,
            "max_rounds": state.max_rounds,
            "score": state.score,
            "grid": {
                "width": state.grid.width,
                "height": state.grid.height,
                "walls": [list(w) for w in sorted(state.grid.walls)],
            },
            "shelves": self._shelf_positions,
            "drop_off": list(state.drop_off),
            "bots": bots,
            "items": items,
            "orders": orders,
            "one_way": self._one_way,
        }

    def _run_server(self) -> None:
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            from websockets.asyncio.server import serve

            async def run():
                async with serve(
                    self._handler, "localhost", self._port,
                    max_size=10 * 1024 * 1024,  # 10 MB for replay data
                ):
                    logger.info("Viz WS server listening on port %d", self._port)
                    await asyncio.Future()  # run forever

            self._loop.run_until_complete(run())
        except Exception:
            logger.exception("Viz broadcaster failed")
        finally:
            self._loop.close()

    async def _handler(self, websocket) -> None:
        """Handle new WebSocket connection."""
        self._clients.add(websocket)
        print(f"[VIZ] Client connected ({len(self._clients)} total), "
              f"replay states: {len(self._all_states)}")
        try:
            # Send full replay history for sim mode (client plays it back)
            if self._all_states:
                replay = json.dumps({
                    "type": "replay",
                    "states": self._all_states,
                })
                print(f"[VIZ] Sending replay: {len(replay):,} bytes")
                await websocket.send(replay)
                print("[VIZ] Replay sent OK")
            elif self._last_state:
                await websocket.send(json.dumps(self._last_state))
            # Keep connection alive
            async for _ in websocket:
                pass  # We don't expect messages from client
        except Exception as e:
            print(f"[VIZ] Handler error: {e}")
        finally:
            self._clients.discard(websocket)
            print(f"[VIZ] Client disconnected ({len(self._clients)} remaining)")

    async def _broadcast(self, msg: str) -> None:
        """Send message to all connected clients."""
        if not self._clients:
            return
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead

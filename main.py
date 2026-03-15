"""
WebSocket client — connects to game server and runs the bot.

Thin layer: receive JSON, pass to Coordinator, send response.
All game logic lives in the bot package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import websockets

try:
    import scipy.optimize  # noqa: F401
except ImportError:
    pass

from bot.coordinator import Coordinator
from bot.config import CoordinatorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run(ws_url: str, config: CoordinatorConfig | None = None, viz: bool = False) -> None:
    """Main game loop — connect, receive, decide, respond."""
    coordinator = Coordinator(config=config)
    if viz:
        coordinator.start_viz()

    logger.info("Connecting to %s ...", ws_url)

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type")

            # Drain buffered messages to prevent desync after network delays.
            # If multiple game_states arrived while we were processing, skip to
            # the latest one. This prevents a permanent 1-round offset where our
            # actions get applied to the wrong round.
            while True:
                try:
                    buffered = await asyncio.wait_for(ws.recv(), timeout=0.005)
                    buffered_data = json.loads(buffered)
                    if buffered_data.get("type") == "game_over":
                        data = buffered_data
                        msg_type = "game_over"
                        break
                    if buffered_data.get("type") == "game_state":
                        skipped_round = data.get("round", "?")
                        data = buffered_data
                        msg_type = "game_state"
                        logger.warning("Draining stale round %s, jumping to round %s",
                                       skipped_round, data.get("round", "?"))
                except (asyncio.TimeoutError, TimeoutError):
                    break

            if msg_type == "game_state":
                response = coordinator.on_game_state(data)
                await ws.send(json.dumps(response))

            elif msg_type == "game_over":
                score = data.get("score", "?")
                rounds_used = data.get("rounds_used", "?")
                items_delivered = data.get("items_delivered", "?")
                orders_completed = data.get("orders_completed", "?")
                logger.info(
                    "Game over! Score: %s | Rounds: %s | Items: %s | Orders: %s",
                    score, rounds_used, items_delivered, orders_completed,
                )
                recon = coordinator._game_logger.finalize(
                    data.get("rounds_used", 300), data.get("score", 0),
                )
                fp = recon.get("fingerprint", "unknown")
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)

                # Save recon — don't overwrite if existing file has more orders
                recon_path = logs_dir / f"{fp}_{date.today()}_recon.json"
                should_save_recon = True
                if recon_path.exists():
                    try:
                        existing = json.loads(recon_path.read_text(encoding="utf-8"))
                        existing_orders = len(existing.get("order_sequence", []))
                        new_orders = len(recon.get("order_sequence", []))
                        if existing_orders > new_orders:
                            logger.info(
                                "Keeping existing recon (%d orders > %d new)",
                                existing_orders, new_orders,
                            )
                            should_save_recon = False
                    except Exception:
                        pass
                if should_save_recon:
                    recon_path.write_text(json.dumps(recon, indent=2), encoding="utf-8")
                    logger.info("Recon saved to %s", recon_path)
                else:
                    # Always keep recon from this run for sim/debug (e.g. score52_recon.json)
                    final_score = data.get("score", 0)
                    score_recon_path = logs_dir / f"{fp}_{date.today()}_score{final_score}_recon.json"
                    score_recon_path.write_text(json.dumps(recon, indent=2), encoding="utf-8")
                    logger.info("Recon saved to %s (score run)", score_recon_path)

                # Auto-generate plan — skip if plan already exists (replay run)
                plan_path = logs_dir / f"{fp}_{date.today()}_plan.json"
                if plan_path.exists():
                    logger.info("Plan already exists at %s, skipping regeneration", plan_path)
                else:
                    try:
                        from bot.engine.pathfinding import PathEngine
                        from bot.recon.analyzer import OfflinePlanner
                        from bot.models import Grid

                        pe = PathEngine()
                        grid_size = recon.get("grid_size", [0, 0])
                        walls = frozenset(tuple(w) for w in recon.get("walls", []))
                        shelves = frozenset(
                            tuple(p) for positions in recon.get("shelf_map", {}).values()
                            for p in positions
                        )
                        grid = Grid(grid_size[0], grid_size[1], walls | shelves)
                        pe.set_grid(grid)

                        planner = OfflinePlanner(recon, pe)
                        game_plan = planner.plan()
                        plan_path.write_text(json.dumps(game_plan, indent=2), encoding="utf-8")
                        logger.info("Plan generated and saved to %s", plan_path)
                    except Exception:
                        logger.exception("Failed to generate plan from recon")
                coordinator.finalize_game(
                    total_rounds=data.get("rounds_used", 300),
                    final_score=data.get("score", 0),
                )

                # Save viz replay if --viz was used
                if viz and coordinator._viz and coordinator._viz._all_states:
                    replay_path = Path("viz/public/replay.json")
                    replay_path.parent.mkdir(parents=True, exist_ok=True)
                    replay_path.write_text(
                        json.dumps({"type": "replay", "states": coordinator._viz._all_states}),
                        encoding="utf-8",
                    )
                    logger.info("Viz replay saved (%d states) — open http://localhost:3000",
                                len(coordinator._viz._all_states))

                coordinator.reset()
                break

            elif msg_type == "error":
                logger.error("Server error: %s", data.get("message", data))

            else:
                logger.debug("Unknown message type: %s", msg_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grocery Bot — NM i AI")
    parser.add_argument(
        "--url",
        default=os.environ.get("GAME_WS_URL", ""),
        help="WebSocket URL (including token), e.g. wss://game-dev.ainm.no/ws?token=...",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config (e.g. logs/best_config.json from optimizer). If set, overrides for_difficulty.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Start visualization WebSocket server on port 8765",
    )
    args = parser.parse_args()

    if not args.url:
        print("Error: No WebSocket URL provided.")
        print("Usage: py main.py --url \"wss://game-dev.ainm.no/ws?token=<JWT>\"")
        print("  or set GAME_WS_URL environment variable.")
        sys.exit(1)

    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                config = CoordinatorConfig.from_dict(data)
                logger.info("Loaded config from %s", config_path)
            except Exception:
                logger.exception("Failed to load config from %s, using default", config_path)
        else:
            logger.warning("Config file %s not found, using default", config_path)

    try:
        asyncio.run(run(args.url, config=config, viz=getattr(args, 'viz', False)))
    except KeyboardInterrupt:
        logger.info("Shutting down")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()

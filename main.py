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

import websockets

from bot.coordinator import Coordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run(ws_url: str) -> None:
    """Main game loop — connect, receive, decide, respond."""
    coordinator = Coordinator()

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
                coordinator.finalize_game(
                    total_rounds=data.get("rounds_used", 300),
                    final_score=data.get("score", 0),
                )
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
    args = parser.parse_args()

    if not args.url:
        print("Error: No WebSocket URL provided.")
        print("Usage: py main.py --url \"wss://game-dev.ainm.no/ws?token=<JWT>\"")
        print("  or set GAME_WS_URL environment variable.")
        sys.exit(1)

    try:
        asyncio.run(run(args.url))
    except KeyboardInterrupt:
        logger.info("Shutting down")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()

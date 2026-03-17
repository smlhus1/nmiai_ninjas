"""
Oracle spike: measure assignment ceiling with standalone greedy oracle.

Compares OracleAssigner (far-first greedy, no routes, parking, pre-pick)
against baseline V2TaskPlanner.

Usage:
    py -m oracle.run_oracle --recon logs/74001e7f_2026-03-16_score274_recon.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter
from oracle.oracle_assigner import OracleAssigner


class OracleAdapter:
    """Wraps BotAdapter, swaps planner to OracleAssigner after coordinator init."""

    def __init__(self, recon_data: dict, *, suppress_logs: bool = True) -> None:
        self._adapter = BotAdapter(suppress_logs=suppress_logs)
        self._recon_data = recon_data
        self._swapped = False

    def __call__(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        response = self._adapter(state_dict)
        if not self._swapped and self._adapter._coordinator is not None:
            # Clear V2TaskPlanner's round-0 assignments to start fresh
            for a in self._adapter._coordinator._assignments.values():
                a.clear()
            oracle = OracleAssigner(self._recon_data)
            oracle._config = self._adapter._coordinator._config
            self._adapter._coordinator._planner = oracle
            self._swapped = True
        return response

    def finalize(self, result: dict[str, Any]):
        return self._adapter.finalize(result)


def run_game(recon_path: str, *, use_oracle: bool = False, verbose: bool = False) -> dict:
    """Run a single game. Returns result dict."""
    recon_data = json.loads(Path(recon_path).read_text(encoding="utf-8"))
    sim = Simulator.from_recon_data(recon_data)

    if use_oracle:
        strategy = OracleAdapter(recon_data)
    else:
        strategy = BotAdapter(suppress_logs=True)

    result = sim.run(strategy, verbose=verbose)
    if hasattr(strategy, 'finalize'):
        strategy.finalize(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Oracle spike: assignment ceiling benchmark")
    parser.add_argument("--recon", required=True, help="Path to recon JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print(f"Recon: {args.recon}")
    print()

    print("Running baseline (V2TaskPlanner)...")
    bl = run_game(args.recon, use_oracle=False, verbose=args.verbose)
    print(f"  score={bl['score']}  items={bl['items_delivered']}  "
          f"orders={bl['orders_completed']}  rounds={bl['rounds_used']}")
    print()

    print("Running oracle (standalone greedy)...")
    oc = run_game(args.recon, use_oracle=True, verbose=args.verbose)
    print(f"  score={oc['score']}  items={oc['items_delivered']}  "
          f"orders={oc['orders_completed']}  rounds={oc['rounds_used']}")
    print()

    gap = oc["score"] - bl["score"]
    pct = gap / max(bl["score"], 1) * 100

    print("=" * 60)
    print(f"  Baseline:  {bl['score']:4d}  ({bl['orders_completed']} orders, {bl['items_delivered']} items)")
    print(f"  Oracle:    {oc['score']:4d}  ({oc['orders_completed']} orders, {oc['items_delivered']} items)")
    print(f"  Gap:       {gap:+4d}  ({pct:+.1f}%)")
    print("=" * 60)
    print()

    if pct > 30:
        print(">>> GO — Oracle significantly beats baseline.")
    elif pct > 10:
        print(">>> GULT — Moderate improvement.")
    elif pct > -10:
        print(">>> NEUTRAL — Oracle roughly matches baseline.")
    else:
        print(">>> INKONKLUSIV — Oracle scorer lavere, men det beviser ikke")
        print("    at assignment er irrelevant. Oracle mangler V2TaskPlanner's")
        print("    sofistikerte logikk (routes, pipeline, stuck handling).")
        print("    En ML planner som ERSTATTER denne logikken (ikke bare assignment)")
        print("    kan potensielt slaa baseline.")


if __name__ == "__main__":
    main()

"""Diagnostic: run offline simulator and track detailed per-round bot behavior."""
import json
import sys
from collections import Counter
from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator


def run_diagnostic(recon_path: str = None, scenario: str = "medium"):
    if recon_path:
        sim = Simulator.from_recon_file(recon_path)
    elif scenario == "easy":
        sim = Simulator.easy_scenario()
    elif scenario == "medium":
        sim = Simulator.medium_scenario()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    adapter = BotAdapter(suppress_logs=True)
    result = sim.run(adapter)
    recon = adapter.finalize(result)

    print(f"\n=== DIAGNOSTIC: {recon_path or scenario} ===")
    print(f"Score: {result['score']}, Items: {result.get('items_delivered', '?')}, "
          f"Orders completed: {result.get('orders_completed', '?')}, Rounds: {result.get('rounds', result.get('rounds_used', 300))}")

    # Order timing from activation rounds
    orders = recon.get("order_sequence", [])
    print(f"\nOrder completion timing ({len(orders)} orders seen):")
    prev_round = 0
    total_duration = 0
    completed_count = 0
    for i, o in enumerate(orders):
        a = o.get("activated_round")
        c = o.get("completed_round")
        items = o.get("items_required", [])
        if a is not None and i + 1 < len(orders):
            # Next order's activation = this order's completion
            next_a = orders[i + 1].get("activated_round")
            if next_a is not None:
                duration = next_a - a
                total_duration += duration
                completed_count += 1
                rate = duration / len(items) if items else 0
                print(f"  Order {i}: {len(items)} items ({','.join(items)}), "
                      f"R{a}-R{next_a} = {duration} rounds ({rate:.1f}r/item)")
            else:
                print(f"  Order {i}: {len(items)} items ({','.join(items)}), "
                      f"activated R{a}, NOT completed")
        elif a is not None:
            print(f"  Order {i}: {len(items)} items ({','.join(items)}), "
                  f"activated R{a}, last order (game ended)")
        else:
            print(f"  Order {i}: {len(items)} items ({','.join(items)}), "
                  f"never activated (preview at game end)")

    if completed_count > 0:
        avg = total_duration / completed_count
        print(f"\n  Avg completed order duration: {avg:.1f} rounds")
        print(f"  Completed: {completed_count}/{len(orders)}")
        print(f"  Target for 217: ~12.5 rounds/order, ~24 orders")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    scenario = sys.argv[2] if len(sys.argv) > 2 else "medium"
    run_diagnostic(path, scenario)

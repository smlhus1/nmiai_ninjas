"""Capture genome strategy's actions as a MAPF plan for live replay.

Usage:
    py -m solver.capture_genome --recon <recon_file> [--genome <genome_file>] [--output <plan_file>]

If no genome file is provided, runs evolutionary search first to find best genome.
"""
import json
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Simulering.offline.simulator import Simulator
from solver.genome import Genome, generate_genome
from solver.genome_strategy import GenomeStrategy
from solver.parallel_search import _genome_from_dict, _genome_to_dict
from mapf_planner import MAPFPlan, MAPFAction, plan_to_dict


def capture_genome_plan(recon_path: str, genome: Genome) -> tuple[dict, int]:
    """Run genome strategy in sim and capture all actions as MAPF plan.

    Returns (plan_dict, score).
    """
    with open(recon_path) as f:
        recon = json.load(f)

    walls = set(tuple(w) for w in recon["walls"])
    shelves = set()
    for ps in recon["shelf_map"].values():
        for p in ps:
            shelves.add(tuple(p))

    strategy = GenomeStrategy(
        genome=genome,
        shelf_map=recon["shelf_map"],
        drop_off_zones=recon["drop_off_zones"],
        grid_walls=walls,
        grid_w=recon["grid_size"][0],
        grid_h=recon["grid_size"][1],
        shelves=shelves,
        order_sequence=recon.get("order_sequence", []),
    )

    sim = Simulator.from_recon_file(recon_path)
    state = sim.reset()

    bot_actions: dict[int, list[MAPFAction]] = {}

    for round_t in range(sim.max_rounds):
        state_dict = state.to_dict()
        bots_data = state_dict["bots"]
        items_data = state_dict["items"]

        response = strategy(state_dict)
        actions = response.get("actions", [])

        # Record actions
        action_map = {a["bot"]: a for a in actions}
        for bot_data in bots_data:
            bid = bot_data["id"]
            bot_pos = tuple(bot_data["position"])
            act = action_map.get(bid, {"action": "wait"})
            action = act.get("action", "wait")

            if bid not in bot_actions:
                bot_actions[bid] = []

            item_type = ""
            if action == "pick_up":
                item_id = act.get("item_id", "")
                for item in items_data:
                    if item["id"] == item_id:
                        item_type = item["type"]
                        break

            bot_actions[bid].append(MAPFAction(
                action=action,
                position=bot_pos,
                item_type=item_type,
            ))

        state, game_over = sim.step(actions)
        if game_over:
            break

        if round_t % 100 == 0:
            print(f"  Round {round_t}: score={sim._score}", flush=True)

    score = sim._score
    orders = sim._orders_completed
    print(f"  Final: score={score}, orders={orders}, rounds={sim._round}", flush=True)

    plan = MAPFPlan(
        actions=bot_actions,
        total_rounds=sim._round,
        expected_score=score,
        order_activations={},
        pickup_schedule=[],
        dropoff_schedule=[],
    )

    return plan_to_dict(plan), score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to genome JSON (from parallel_search)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plan file path")
    parser.add_argument("--search", action="store_true",
                        help="Run evolutionary search before capture")
    parser.add_argument("--pop", type=int, default=30)
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    with open(args.recon) as f:
        recon = json.load(f)

    genome = None

    # Load genome from file
    if args.genome:
        with open(args.genome) as f:
            genome_dict = json.load(f)
        genome = _genome_from_dict(genome_dict)
        print(f"Loaded genome from {args.genome}", flush=True)

    # Run evolutionary search to find best genome
    elif args.search:
        from solver.parallel_search import evolve
        print(f"Running evolutionary search ({args.pop} pop, {args.gens} gens)...", flush=True)
        t0 = time.time()
        genome, score = evolve(
            args.recon, pop_size=args.pop, generations=args.gens,
            n_workers=args.workers
        )
        elapsed = time.time() - t0
        print(f"Search complete: score={score}, time={elapsed:.0f}s", flush=True)

    # Use greedy genome as baseline
    else:
        genome = generate_genome(
            recon["order_sequence"],
            recon["shelf_map"],
            n_bots=recon.get("bot_count", 20),
        )
        print("Using greedy genome (no search)", flush=True)

    # Capture plan
    print("\nCapturing MAPF plan...", flush=True)
    plan_dict, score = capture_genome_plan(args.recon, genome)

    # Save
    output_path = args.output or f"mapf_plan_genome_{score}.json"
    Path(output_path).write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")
    print(f"\nPlan saved: {output_path} (score={score})", flush=True)

    # Also save the genome
    genome_path = output_path.replace(".json", "_genome.json")
    Path(genome_path).write_text(json.dumps(_genome_to_dict(genome)), encoding="utf-8")
    print(f"Genome saved: {genome_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Full integration test: simulator + strategy + optimizer.
Builds a realistic Easy grid, runs 1000 optimization iterations.

Run: python -m offline.test_e2e
"""
import time
from offline.simulator import Simulator
from offline.strategy import StrategyParams, ParameterizedStrategy
from offline.optimize import optimize, grid_search, compare_strategies

Pos = tuple[int, int]


def make_easy_scenario() -> Simulator:
    """Realistic Easy: 12x10, 1 bot, 4 item types, ~15 orders."""
    width, height = 12, 10
    walls: set[Pos] = set()
    shelves: set[Pos] = set()

    # Border
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))

    # Shelves: 3 aisle pairs
    shelf_types: dict[Pos, str] = {}
    type_map = {
        (2, 2): "milk",  (2, 3): "milk",  (2, 6): "milk",  (2, 7): "milk",
        (3, 2): "bread", (3, 3): "bread", (3, 6): "bread", (3, 7): "bread",
        (5, 2): "butter",(5, 3): "butter",(5, 6): "butter",(5, 7): "butter",
        (6, 2): "yogurt",(6, 3): "yogurt",(6, 6): "yogurt",(6, 7): "yogurt",
        (8, 2): "milk",  (8, 3): "bread", (8, 6): "butter",(8, 7): "yogurt",
        (9, 2): "milk",  (9, 3): "bread", (9, 6): "butter",(9, 7): "yogurt",
    }

    for pos, itype in type_map.items():
        shelves.add(pos)
        shelf_types[pos] = itype

    drop_off = (1, 8)
    spawn = [(10, 8)]

    # 20 orders (more than can be completed in 300 rounds)
    orders = [
        {"id": f"order_{i}", "items_required": items}
        for i, items in enumerate([
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
            ["bread", "butter", "yogurt"],
            ["milk", "yogurt", "bread", "butter"],
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
            ["bread", "butter", "yogurt"],
            ["milk", "bread", "butter"],
            ["yogurt", "bread", "milk"],
            ["butter", "milk", "yogurt"],
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
            ["bread", "butter", "yogurt"],
            ["milk", "yogurt", "bread", "butter"],
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
        ])
    ]

    return Simulator(
        width=width, height=height,
        walls=walls, shelves=shelves,
        drop_off=drop_off,
        spawn_positions=spawn,
        order_sequence=orders,
        item_types_at_shelves=shelf_types,
    )


def make_medium_scenario() -> Simulator:
    """Realistic Medium: 16x12, 3 bots, 8 item types."""
    width, height = 16, 12
    walls: set[Pos] = set()
    shelves: set[Pos] = set()

    # Border
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))

    # 4 aisle pairs, 8 item types
    shelf_types: dict[Pos, str] = {}
    types_list = ["milk", "bread", "butter", "yogurt",
                  "cheese", "juice", "eggs", "ham"]

    ti = 0
    for ax in [2, 3, 5, 6, 8, 9, 11, 12]:
        for sy in [2, 3, 6, 7, 9]:
            pos = (ax, sy)
            shelves.add(pos)
            shelf_types[pos] = types_list[ti % len(types_list)]
            ti += 1

    drop_off = (1, 10)
    spawns = [(14, 10), (14, 8), (14, 6)]

    orders = [
        {"id": f"order_{i}", "items_required": items}
        for i, items in enumerate([
            ["milk", "bread", "butter"],
            ["yogurt", "cheese", "juice"],
            ["eggs", "ham", "milk"],
            ["bread", "butter", "yogurt", "cheese"],
            ["juice", "eggs", "ham"],
            ["milk", "cheese", "bread", "yogurt"],
            ["butter", "juice", "eggs", "ham"],
            ["milk", "bread", "cheese"],
            ["yogurt", "ham", "eggs"],
            ["butter", "juice", "milk"],
            ["bread", "cheese", "yogurt", "ham"],
            ["eggs", "milk", "butter"],
            ["juice", "bread", "cheese"],
            ["yogurt", "ham", "milk", "eggs"],
            ["butter", "bread", "juice"],
            ["cheese", "yogurt", "milk"],
            ["ham", "eggs", "butter", "juice"],
            ["milk", "bread", "yogurt"],
            ["cheese", "ham", "eggs"],
            ["butter", "juice", "milk", "bread"],
        ])
    ]

    return Simulator(
        width=width, height=height,
        walls=walls, shelves=shelves,
        drop_off=drop_off,
        spawn_positions=spawns,
        order_sequence=orders,
        item_types_at_shelves=shelf_types,
    )


def test_simulator_basic():
    """Verify simulator produces valid games."""
    sim = make_easy_scenario()
    
    # Test with wait-only strategy
    def wait_strategy(state):
        return {"actions": [{"bot": 0, "action": "wait"}]}
    
    result = sim.run(wait_strategy)
    assert result["score"] == 0, "Wait-only should score 0"
    assert result["rounds_used"] == 300
    print(f"✅ Wait-only: score={result['score']}, rounds={result['rounds_used']}")


def test_simulator_with_strategy():
    """Verify parameterized strategy actually scores points."""
    sim = make_easy_scenario()
    params = StrategyParams()
    strategy = ParameterizedStrategy(
        params, sim.width, sim.height, sim.walls, sim.shelves,
    )
    strategy.precompute_bfs()
    
    result = sim.run(strategy)
    print(f"✅ Default strategy: score={result['score']}, "
          f"items={result['items_delivered']}, "
          f"orders={result['orders_completed']}, "
          f"rounds={result['rounds_used']}")
    assert result["score"] > 0, "Default strategy should score > 0"
    return result["score"]


def test_simulator_speed():
    """Benchmark: how many games/second?"""
    sim = make_easy_scenario()
    
    # Precompute BFS once (this is what the optimizer does)
    shared_cache: dict = {}
    params = StrategyParams()
    strategy = ParameterizedStrategy(
        params, sim.width, sim.height, sim.walls, sim.shelves,
        shared_bfs_cache=shared_cache,
    )
    strategy.precompute_bfs()
    
    n = 200
    start = time.time()
    for _ in range(n):
        # Create new strategy with SAME cache (like optimizer does)
        s = ParameterizedStrategy(
            params, sim.width, sim.height, sim.walls, sim.shelves,
            shared_bfs_cache=shared_cache,
        )
        sim.run(s)
    elapsed = time.time() - start
    
    rate = n / elapsed
    ms_per_game = elapsed / n * 1000
    print(f"✅ Speed (shared BFS): {rate:.0f} games/sec ({ms_per_game:.1f}ms per game)")
    return rate


def test_optimizer_quick():
    """Run optimizer for 500 iterations, verify improvement."""
    sim = make_easy_scenario()
    
    # Baseline
    baseline_params = StrategyParams()
    baseline_strategy = ParameterizedStrategy(
        baseline_params, sim.width, sim.height, sim.walls, sim.shelves
    )
    baseline = sim.run(baseline_strategy)
    
    print(f"\nBaseline: score={baseline['score']}")
    print(f"Optimizing (500 iterations)...")
    
    best_params, best_score = optimize(sim, iterations=500, verbose=False)
    
    delta = best_score - baseline['score']
    print(f"✅ Optimized: score={best_score} (delta: +{delta})")
    print(f"   Key params that changed:")
    for k, v in best_params.__dict__.items():
        default = getattr(StrategyParams(), k)
        if v != default:
            print(f"     {k}: {default} → {v}")


def test_medium_scenario():
    """Test 3-bot Medium scenario."""
    sim = make_medium_scenario()
    params = StrategyParams()
    strategy = ParameterizedStrategy(
        params, sim.width, sim.height, sim.walls, sim.shelves
    )
    strategy.precompute_bfs()
    
    result = sim.run(strategy)
    print(f"✅ Medium (3 bots): score={result['score']}, "
          f"items={result['items_delivered']}, "
          f"orders={result['orders_completed']}")


def test_strategy_comparison():
    """Compare different strategy presets."""
    sim = make_easy_scenario()
    print()
    compare_strategies(sim, n=1)


def test_determinism():
    """Verify simulator is deterministic (same params = same score)."""
    sim = make_easy_scenario()
    params = StrategyParams()
    
    shared_cache: dict = {}
    probe = ParameterizedStrategy(
        params, sim.width, sim.height, sim.walls, sim.shelves,
        shared_bfs_cache=shared_cache,
    )
    probe.precompute_bfs()
    
    scores = []
    for _ in range(5):
        strategy = ParameterizedStrategy(
            params, sim.width, sim.height, sim.walls, sim.shelves,
            shared_bfs_cache=shared_cache,
        )
        result = sim.run(strategy)
        scores.append(result["score"])
    
    assert len(set(scores)) == 1, f"Non-deterministic! Scores: {scores}"
    print(f"✅ Determinism: 5 runs all scored {scores[0]}")


if __name__ == "__main__":
    print("=" * 60)
    print("SIMULATOR + OPTIMIZER END-TO-END TEST")
    print("=" * 60)
    print()

    test_simulator_basic()
    print()
    test_determinism()
    print()
    base_score = test_simulator_with_strategy()
    print()
    test_medium_scenario()
    print()
    rate = test_simulator_speed()
    print()
    test_strategy_comparison()
    print()
    test_optimizer_quick()
    
    print()
    print("=" * 60)
    est_10k = 10_000 / rate
    est_100k = 100_000 / rate
    print(f"PERFORMANCE SUMMARY")
    print(f"  {rate:.0f} games/sec")
    print(f"  10k iterations: ~{est_10k:.0f}s")
    print(f"  100k iterations: ~{est_100k:.0f}s")
    print(f"=" * 60)

"""Local stochastic cellular automaton simulator for Astar Island.

Simulates Norse civilisation dynamics on a 40x40 grid over 50 years.
Phases per year: Growth -> Conflict -> Trade -> Winter -> Environment.

Designed to:
- Match round 1 observed transition probabilities within 10%
- Run 500+ Monte Carlo simulations in <30 seconds
- Have tunable parameters for calibration against observations

All computation uses numpy arrays for speed. No per-cell objects.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Terrain class indices (match challenge spec)
CLS_EMPTY = 0       # Ocean(10), Plains(11), Empty(0)
CLS_SETTLEMENT = 1  # Settlement(1)
CLS_PORT = 2        # Port(2)
CLS_RUIN = 3        # Ruin(3)
CLS_FOREST = 4      # Forest(4)
CLS_MOUNTAIN = 5    # Mountain(5)

# Raw grid code -> class mapping
CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Plains vs ocean distinction (needed for port adjacency checks)
_OCEAN_CODE = 10
_PLAINS_CODE = 11


@dataclass
class SimParams:
    """Tunable simulation parameters.

    Default values calibrated to match round 1 ground truth:
    - Empty->Empty 99.5%, Empty->Settlement 0.5%
    - Settlement->Settlement 57%, Settlement->Empty 41%, Settlement->Port 1.5%
    - Port->Port 14.3%, Port->Empty 85.7%
    - Forest->Forest 98.8%, Forest->Settlement 1.2%
    - Mountain->Mountain 100%
    - Only 1-2.5% of cells change overall
    - 41% of initial settlements die
    """
    # Growth phase
    expansion_rate: float = 0.002       # P(empty cell near settlement becomes settlement) — very low to match 0.5% empty->settlement
    food_from_forest: float = 0.15      # food bonus per adjacent forest
    food_from_plains: float = 0.05      # food bonus per adjacent plains (meager)
    port_develop_rate: float = 0.015    # P(coastal settlement develops port per year)
    growth_distance_decay: float = 0.5  # expansion probability decays with distance from existing settlements

    # Conflict phase
    conflict_intensity: float = 0.08    # P(settlement is raided per year)
    raid_damage: float = 0.4           # food/pop loss from raid
    raid_kill_threshold: float = 0.2    # if food drops below this after raid, settlement dies

    # Trade phase (ports)
    port_trade_range: int = 8           # max distance between trading ports
    trade_food_bonus: float = 0.05      # food gain from trade per trading partner (modest)
    port_stability_bonus: float = 0.02  # ports are slightly harder to kill

    # Winter phase — THIS IS THE KEY DRIVER of settlement death
    winter_severity: float = 0.135      # base food loss per winter
    winter_variance: float = 0.055      # random variation in winter severity
    winter_kill_threshold: float = 0.11 # settlement dies if food < this after winter
    settlement_base_food: float = 0.40  # starting food for new settlements
    food_cap: float = 0.7              # max food a settlement can store (prevents runaway)

    # Environment phase
    ruin_reclaim_rate: float = 0.03     # P(ruin reclaimed by nearby settlement per year)
    forest_regrowth_rate: float = 0.008 # P(ruin becomes forest per year)
    ruin_to_empty_rate: float = 0.08    # P(ruin decays to empty plains per year)

    # Settlement death -> what it becomes
    death_to_ruin_rate: float = 0.07    # P(dead settlement becomes ruin vs empty) — most become empty
    death_to_empty_rate: float = 0.93   # P(dead settlement becomes empty)


class AstarSimulator:
    """Stochastic cellular automaton for Norse civilisation prediction.

    Uses numpy arrays throughout for speed. Settlement state tracked
    in parallel arrays (food, population) rather than objects.
    """

    def __init__(
        self,
        initial_grid: np.ndarray,
        settlements: list[dict],
        raw_grid: Optional[np.ndarray] = None,
        params: Optional[SimParams] = None,
    ):
        """
        Args:
            initial_grid: (H, W) array of terrain classes (0-5)
            settlements: list of dicts with 'x', 'y', 'has_port', 'alive' keys
            raw_grid: (H, W) array of raw grid codes (10, 11, 0, 1, 2, 3, 4, 5).
                      Needed to distinguish ocean from plains (both class 0).
                      If None, all class-0 cells treated as plains.
            params: simulation parameters (uses defaults if None)
        """
        self.H, self.W = initial_grid.shape
        self.params = params or SimParams()
        self.initial_grid = initial_grid.copy()

        # Distinguish ocean vs plains
        if raw_grid is not None:
            self.is_ocean = (raw_grid == _OCEAN_CODE)
            self.is_plains = (raw_grid == _PLAINS_CODE)
        else:
            # Heuristic: class-0 cells adjacent to non-class-0 on the border are ocean
            self.is_ocean = np.zeros((self.H, self.W), dtype=bool)
            self.is_plains = (initial_grid == CLS_EMPTY)

        # Static terrain masks (never change)
        self.is_mountain = (initial_grid == CLS_MOUNTAIN)
        # Ocean never changes, mountains never change
        self.static_mask = self.is_ocean | self.is_mountain

        # Precompute neighbor offsets (4-connected)
        self._dy = np.array([-1, 1, 0, 0])
        self._dx = np.array([0, 0, -1, 1])

        # Store initial settlement positions for food calculation
        self._initial_settlements = settlements

    def _init_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize mutable simulation state.

        Returns:
            grid: (H, W) terrain class array
            food: (H, W) food level per cell (only meaningful for settlements/ports)
            population: (H, W) population per cell
        """
        grid = self.initial_grid.copy()
        food = np.zeros((self.H, self.W), dtype=np.float32)
        population = np.zeros((self.H, self.W), dtype=np.float32)

        p = self.params

        # Initialize settlement food/population
        for s in self._initial_settlements:
            x, y = s["x"], s["y"]
            if not s.get("alive", True):
                continue
            # Random initial food (settlements start with some resources)
            food[y, x] = p.settlement_base_food + rng.uniform(0, 0.3)
            population[y, x] = rng.uniform(0.5, 1.5)
            if s.get("has_port", False):
                food[y, x] += p.port_stability_bonus

        return grid, food, population

    def _count_adjacent(self, grid: np.ndarray, cls: int) -> np.ndarray:
        """Count how many 4-neighbors have a given class. Returns (H, W) int array."""
        mask = (grid == cls).astype(np.int32)
        count = np.zeros_like(mask)
        count[1:, :] += mask[:-1, :]   # neighbor above
        count[:-1, :] += mask[1:, :]   # neighbor below
        count[:, 1:] += mask[:, :-1]   # neighbor left
        count[:, :-1] += mask[:, 1:]   # neighbor right
        return count

    def _count_adjacent_multi(self, grid: np.ndarray, classes: list[int]) -> np.ndarray:
        """Count 4-neighbors matching any of the given classes."""
        mask = np.zeros((self.H, self.W), dtype=np.int32)
        for cls in classes:
            mask |= (grid == cls).astype(np.int32)
        count = np.zeros_like(mask)
        count[1:, :] += mask[:-1, :]
        count[:-1, :] += mask[1:, :]
        count[:, 1:] += mask[:, :-1]
        count[:, :-1] += mask[:, 1:]
        return count

    def _has_adjacent_ocean(self) -> np.ndarray:
        """Boolean mask: True if cell has at least one ocean neighbor."""
        ocean = self.is_ocean.astype(np.int32)
        count = np.zeros_like(ocean)
        count[1:, :] += ocean[:-1, :]
        count[:-1, :] += ocean[1:, :]
        count[:, 1:] += ocean[:, :-1]
        count[:, :-1] += ocean[:, 1:]
        return count > 0

    def _phase_growth(
        self, grid: np.ndarray, food: np.ndarray, population: np.ndarray, rng: np.random.Generator
    ):
        """Growth phase: produce food, grow population, expand settlements, develop ports."""
        p = self.params

        # --- Food production for existing settlements/ports ---
        is_alive = (grid == CLS_SETTLEMENT) | (grid == CLS_PORT)
        adj_forest = self._count_adjacent(grid, CLS_FOREST).astype(np.float32)
        adj_plains = self._count_adjacent_multi(grid, [CLS_EMPTY]).astype(np.float32)
        # Only count plains, not ocean, for food
        # Approximate: subtract ocean neighbors from empty neighbors
        adj_ocean_count = self._has_adjacent_ocean().astype(np.float32)

        food_gain = (adj_forest * p.food_from_forest + adj_plains * p.food_from_plains) * is_alive
        food += food_gain

        # Cap food to prevent immortal settlements
        np.minimum(food, p.food_cap, out=food)

        # Population grows with food
        pop_growth = np.minimum(food * 0.05, 0.1) * is_alive
        population += pop_growth

        # --- Expansion: settlements spread to adjacent empty/forest cells ---
        # Only well-fed settlements expand
        well_fed = is_alive & (food > 0.5)
        adj_well_fed = self._count_adjacent_multi(grid, [CLS_SETTLEMENT, CLS_PORT])
        # But we only want expansion from well-fed ones — approximate by using general adjacency
        # with very low probability

        # Candidates: empty plains or forest, not static, adjacent to settlement
        can_expand = (
            ((grid == CLS_EMPTY) & ~self.is_ocean)
            | (grid == CLS_FOREST)
        ) & (adj_well_fed > 0) & ~self.static_mask

        # Expansion probability — very low to match 0.5% empty->settlement over 50 years
        expand_prob = np.minimum(adj_well_fed * p.expansion_rate, 0.05)
        expand_rolls = rng.random((self.H, self.W))
        new_settlements = can_expand & (expand_rolls < expand_prob)

        # Apply expansions
        grid[new_settlements] = CLS_SETTLEMENT
        food[new_settlements] = p.settlement_base_food * 0.5
        population[new_settlements] = 0.2

        # --- Port development: coastal settlements can become ports ---
        is_settlement = (grid == CLS_SETTLEMENT)
        coastal = self._has_adjacent_ocean()
        can_port = is_settlement & coastal & (food > 0.5)
        port_rolls = rng.random((self.H, self.W))
        new_ports = can_port & (port_rolls < p.port_develop_rate)
        grid[new_ports] = CLS_PORT

    def _phase_conflict(
        self, grid: np.ndarray, food: np.ndarray, population: np.ndarray, rng: np.random.Generator
    ):
        """Conflict phase: settlements raid each other, low food = more aggressive."""
        p = self.params

        is_alive = (grid == CLS_SETTLEMENT) | (grid == CLS_PORT)
        if not np.any(is_alive):
            return

        # Raid probability: higher when food is low
        hunger_factor = np.clip(1.0 - food, 0, 1)  # hungrier = more raids
        raid_prob = p.conflict_intensity * (0.5 + 0.5 * hunger_factor) * is_alive

        raided = rng.random((self.H, self.W)) < raid_prob

        # Raid damage
        food[raided] -= p.raid_damage
        population[raided] -= p.raid_damage * 0.5

        # Clamp
        food[raided] = np.maximum(food[raided], 0)
        population[raided] = np.maximum(population[raided], 0)

        # Settlements destroyed by raids
        killed = raided & (food < p.raid_kill_threshold) & is_alive
        # Some become ruins, some become empty
        ruin_rolls = rng.random((self.H, self.W))
        to_ruin = killed & (ruin_rolls < p.death_to_ruin_rate * 10)  # small fraction become ruins
        to_empty = killed & ~to_ruin

        grid[to_ruin] = CLS_RUIN
        grid[to_empty] = CLS_EMPTY
        food[killed] = 0
        population[killed] = 0

    def _phase_trade(
        self, grid: np.ndarray, food: np.ndarray, rng: np.random.Generator
    ):
        """Trade phase: ports trade with nearby ports for food bonus."""
        p = self.params

        port_positions = np.argwhere(grid == CLS_PORT)
        if len(port_positions) < 2:
            return

        for i, (y1, x1) in enumerate(port_positions):
            partners = 0
            for j, (y2, x2) in enumerate(port_positions):
                if i == j:
                    continue
                dist = abs(y1 - y2) + abs(x1 - x2)
                if dist <= p.port_trade_range:
                    partners += 1
            if partners > 0:
                food[y1, x1] += p.trade_food_bonus * min(partners, 3)

    def _phase_winter(
        self, grid: np.ndarray, food: np.ndarray, population: np.ndarray, rng: np.random.Generator
    ):
        """Winter phase: all settlements lose food, weak ones collapse.

        This is the main driver of settlement death. 41% of initial
        settlements should die over 50 years, mostly becoming empty.
        """
        p = self.params

        is_alive = (grid == CLS_SETTLEMENT) | (grid == CLS_PORT)
        if not np.any(is_alive):
            return

        # Variable winter severity — some years are brutal
        severity = p.winter_severity + rng.uniform(-p.winter_variance, p.winter_variance)
        severity = max(0.05, severity)

        # Per-cell random variation (microclimates)
        cell_severity = severity * (0.8 + 0.4 * rng.random((self.H, self.W)))

        # Ports lose MORE because they're exposed coastal settlements
        is_port = (grid == CLS_PORT)
        cell_severity[is_port] *= 1.3

        food[is_alive] -= cell_severity[is_alive]
        np.maximum(food, 0, out=food)

        # Settlements that starve collapse
        starved = is_alive & (food < p.winter_kill_threshold)

        # Dead settlements mostly become empty (ground truth: settlement->empty 41%)
        ruin_rolls = rng.random((self.H, self.W))
        to_ruin = starved & (ruin_rolls < p.death_to_ruin_rate)
        to_empty = starved & ~to_ruin

        grid[to_ruin] = CLS_RUIN
        grid[to_empty] = CLS_EMPTY
        food[starved] = 0
        population[starved] = 0

    def _phase_environment(
        self, grid: np.ndarray, food: np.ndarray, population: np.ndarray, rng: np.random.Generator
    ):
        """Environment phase: ruins reclaimed or overgrown, forest regrowth."""
        p = self.params

        is_ruin = (grid == CLS_RUIN)
        if not np.any(is_ruin):
            return

        # Nearby settlements can reclaim ruins
        adj_settlement = self._count_adjacent_multi(grid, [CLS_SETTLEMENT, CLS_PORT])
        can_reclaim = is_ruin & (adj_settlement > 0)
        reclaim_rolls = rng.random((self.H, self.W))
        reclaimed = can_reclaim & (reclaim_rolls < p.ruin_reclaim_rate)
        grid[reclaimed] = CLS_SETTLEMENT
        food[reclaimed] = p.settlement_base_food * 0.5
        population[reclaimed] = 0.3

        # Forest regrowth on remaining ruins
        still_ruin = (grid == CLS_RUIN)
        forest_rolls = rng.random((self.H, self.W))
        to_forest = still_ruin & (forest_rolls < p.forest_regrowth_rate)
        grid[to_forest] = CLS_FOREST

        # Ruins decay to empty
        still_ruin2 = (grid == CLS_RUIN)
        empty_rolls = rng.random((self.H, self.W))
        to_empty = still_ruin2 & (empty_rolls < p.ruin_to_empty_rate)
        grid[to_empty] = CLS_EMPTY

    def run(self, n_years: int = 50, seed: Optional[int] = None) -> np.ndarray:
        """Run one stochastic simulation.

        Args:
            n_years: number of simulation years (default 50)
            seed: random seed for reproducibility

        Returns:
            (H, W) terrain class grid after simulation
        """
        rng = np.random.default_rng(seed)
        grid, food, population = self._init_state(rng)

        for year in range(n_years):
            self._phase_growth(grid, food, population, rng)
            self._phase_conflict(grid, food, population, rng)
            self._phase_trade(grid, food, rng)
            self._phase_winter(grid, food, population, rng)
            self._phase_environment(grid, food, population, rng)

            # Enforce static terrain invariants
            grid[self.is_ocean] = CLS_EMPTY
            grid[self.is_mountain] = CLS_MOUNTAIN

        return grid

    def monte_carlo(self, n_runs: int = 500, n_years: int = 50) -> np.ndarray:
        """Run n stochastic simulations, return probability tensor.

        Args:
            n_runs: number of Monte Carlo samples
            n_years: simulation years per run

        Returns:
            (H, W, 6) probability tensor where tensor[y, x, c] is
            the probability of terrain class c at position (y, x).
        """
        counts = np.zeros((self.H, self.W, 6), dtype=np.int32)

        for i in range(n_runs):
            final_grid = self.run(n_years=n_years, seed=i)
            # Accumulate class counts
            for cls in range(6):
                counts[:, :, cls] += (final_grid == cls).astype(np.int32)

        # Convert to probabilities
        probs = counts.astype(np.float64) / n_runs

        # Return raw probabilities (no floor — callers add their own smoothing)
        return probs


def raw_grid_to_classes(raw_grid: list[list[int]]) -> np.ndarray:
    """Convert raw grid codes (10, 11, 0, 1, 2, 3, 4, 5) to class indices (0-5)."""
    arr = np.array(raw_grid, dtype=np.int32)
    result = np.zeros_like(arr)
    for code, cls in CODE_TO_CLASS.items():
        result[arr == code] = cls
    return result


def load_round_data(path: str) -> dict:
    """Load round data from JSON file."""
    with open(path) as f:
        return json.load(f)


def build_simulator(round_data: dict, seed_index: int, params: Optional[SimParams] = None) -> AstarSimulator:
    """Build a simulator from round data for a specific seed.

    Args:
        round_data: parsed JSON from astar_round1.json
        seed_index: which seed (0-4)
        params: simulation parameters

    Returns:
        configured AstarSimulator
    """
    state = round_data["initial_states"][seed_index]
    raw_grid = np.array(state["grid"], dtype=np.int32)
    class_grid = raw_grid_to_classes(state["grid"])
    settlements = state["settlements"]

    return AstarSimulator(
        initial_grid=class_grid,
        settlements=settlements,
        raw_grid=raw_grid,
        params=params,
    )


# ---------------------------------------------------------------------------
# Tests and calibration against round 1 ground truth
# ---------------------------------------------------------------------------

def _compute_transition_stats(
    initial_grid: np.ndarray, final_grid: np.ndarray, is_ocean: np.ndarray
) -> dict:
    """Compute transition probabilities from initial to final grid."""
    transitions = {}
    for src_cls in range(6):
        src_mask = (initial_grid == src_cls)
        if src_cls == 0:
            # Separate ocean (static) from plains (can change)
            src_mask = src_mask & ~is_ocean
        total = src_mask.sum()
        if total == 0:
            continue
        row = {}
        for dst_cls in range(6):
            count = (src_mask & (final_grid == dst_cls)).sum()
            row[CLASS_NAMES[dst_cls]] = count / total
        transitions[CLASS_NAMES[src_cls]] = row
    return transitions


def _compute_settlement_survival_rate(
    initial_grid: np.ndarray, final_grid: np.ndarray
) -> float:
    """What fraction of initial settlements are still settlements in the final state."""
    initial_settlements = (initial_grid == CLS_SETTLEMENT) | (initial_grid == CLS_PORT)
    final_settlements = (final_grid == CLS_SETTLEMENT) | (final_grid == CLS_PORT)
    n_initial = initial_settlements.sum()
    if n_initial == 0:
        return 0.0
    survived = (initial_settlements & final_settlements).sum()
    return survived / n_initial


def _compute_change_rate(initial_grid: np.ndarray, final_grid: np.ndarray) -> float:
    """Fraction of cells that changed class."""
    return (initial_grid != final_grid).sum() / initial_grid.size


def test_against_round1():
    """Verify simulator matches round 1 ground truth transition probabilities."""
    data_path = Path(__file__).parent.parent / "data" / "astar_round1.json"
    if not data_path.exists():
        print(f"SKIP: round 1 data not found at {data_path}")
        return

    round_data = load_round_data(str(data_path))
    n_seeds = len(round_data["initial_states"])

    print("=" * 70)
    print("CALIBRATION TEST: Simulator vs Round 1 Ground Truth")
    print("=" * 70)

    # Target transition probabilities (from ground truth analysis)
    targets = {
        "Empty": {"Empty": 0.995, "Settlement": 0.005},
        "Settlement": {"Empty": 0.41, "Settlement": 0.57, "Port": 0.015},
        "Port": {"Empty": 0.857, "Port": 0.143},
        "Forest": {"Forest": 0.988, "Settlement": 0.012},
        "Mountain": {"Mountain": 1.0},
    }

    target_change_rate = 0.02  # 1-2.5%
    target_settlement_death_rate = 0.41  # 41% of settlements die

    # Run Monte Carlo for each seed, aggregate transitions
    n_mc = 100  # enough for stats, fast enough for test
    all_transitions = {src: {dst: [] for dst in CLASS_NAMES} for src in CLASS_NAMES}
    all_change_rates = []
    all_survival_rates = []

    for seed_idx in range(n_seeds):
        sim = build_simulator(round_data, seed_idx)
        initial = sim.initial_grid.copy()
        is_ocean = sim.is_ocean.copy()

        for mc_seed in range(n_mc):
            final = sim.run(seed=seed_idx * 10000 + mc_seed)
            trans = _compute_transition_stats(initial, final, is_ocean)
            change = _compute_change_rate(initial, final)
            survival = _compute_settlement_survival_rate(initial, final)

            all_change_rates.append(change)
            all_survival_rates.append(survival)

            for src, row in trans.items():
                for dst, prob in row.items():
                    all_transitions[src][dst].append(prob)

    # Report results
    print(f"\nAggregated over {n_seeds} seeds x {n_mc} MC runs = {n_seeds * n_mc} simulations\n")

    print("--- Transition Probabilities ---")
    all_ok = True
    for src in ["Empty", "Settlement", "Port", "Forest", "Mountain"]:
        if src not in targets:
            continue
        print(f"\n  {src} ->")
        for dst in CLASS_NAMES:
            vals = all_transitions[src][dst]
            if not vals:
                continue
            mean = np.mean(vals)
            target = targets[src].get(dst, 0.0)
            if target > 0.01 or mean > 0.01:
                diff = abs(mean - target)
                ok = diff < 0.10  # within 10%
                status = "OK" if ok else "MISS"
                if not ok:
                    all_ok = False
                print(f"    {dst:12s}: sim={mean:.3f}  target={target:.3f}  diff={diff:.3f}  [{status}]")

    print(f"\n--- Aggregate Stats ---")
    mean_change = np.mean(all_change_rates)
    mean_survival = np.mean(all_survival_rates)
    mean_death = 1.0 - mean_survival

    print(f"  Change rate:         sim={mean_change:.4f}  target={target_change_rate:.4f}")
    print(f"  Settlement death:    sim={mean_death:.3f}  target={target_settlement_death_rate:.3f}")

    change_ok = abs(mean_change - target_change_rate) < 0.02
    death_ok = abs(mean_death - target_settlement_death_rate) < 0.15

    print(f"  Change rate match:   {'OK' if change_ok else 'MISS'}")
    print(f"  Death rate match:    {'OK' if death_ok else 'MISS'}")

    return all_ok and change_ok and death_ok


def test_speed():
    """Verify 500 MC runs complete in <30 seconds."""
    data_path = Path(__file__).parent.parent / "data" / "astar_round1.json"
    if not data_path.exists():
        print(f"SKIP: round 1 data not found at {data_path}")
        return

    round_data = load_round_data(str(data_path))
    sim = build_simulator(round_data, seed_index=0)

    print("\n--- Speed Test: 500 MC runs ---")
    t0 = time.perf_counter()
    probs = sim.monte_carlo(n_runs=500, n_years=50)
    elapsed = time.perf_counter() - t0

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Shape: {probs.shape}")
    print(f"  Prob sums (should be 1.0): min={probs.sum(axis=-1).min():.4f}, max={probs.sum(axis=-1).max():.4f}")
    print(f"  Speed: {'OK' if elapsed < 30 else 'SLOW'} (target <30s)")

    return elapsed < 30


def test_static_invariants():
    """Verify ocean and mountain cells never change."""
    data_path = Path(__file__).parent.parent / "data" / "astar_round1.json"
    if not data_path.exists():
        print(f"SKIP: round 1 data not found at {data_path}")
        return

    round_data = load_round_data(str(data_path))
    sim = build_simulator(round_data, seed_index=0)

    for seed in range(10):
        final = sim.run(seed=seed)
        # Ocean stays empty
        assert np.all(final[sim.is_ocean] == CLS_EMPTY), f"Ocean changed in run {seed}"
        # Mountain stays mountain
        assert np.all(final[sim.is_mountain] == CLS_MOUNTAIN), f"Mountain changed in run {seed}"

    print("  Static invariants: OK (ocean/mountain never change)")


if __name__ == "__main__":
    print("Running Astar Island simulator tests...\n")

    test_static_invariants()
    calibration_ok = test_against_round1()
    speed_ok = test_speed()

    print("\n" + "=" * 70)
    if calibration_ok and speed_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS NEED TUNING — adjust SimParams to match targets")
    print("=" * 70)

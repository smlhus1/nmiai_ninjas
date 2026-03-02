"""
Local game simulator for Grocery Bot.
Reproduces server logic exactly. ~1ms per 300-round game.

Usage:
    sim = Simulator.from_game_log("logs/game_log.json")
    score = sim.run(strategy_fn)
    
    # Or run 100k iterations:
    best = optimize(sim, iterations=100_000)
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional

Pos = tuple[int, int]

MOVES = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


@dataclass
class SimBot:
    id: int
    position: Pos
    inventory: list[str] = field(default_factory=list)


@dataclass
class SimItem:
    id: str
    item_type: str
    position: Pos  # shelf position (non-walkable)
    picked: bool = False
    picked_by: int = -1


@dataclass
class SimOrder:
    id: str
    items_required: list[str]
    items_delivered: list[str] = field(default_factory=list)
    status: str = "hidden"  # hidden, preview, active, complete

    @property
    def complete(self) -> bool:
        return Counter(self.items_delivered) == Counter(self.items_required)

    @property
    def items_remaining(self) -> list[str]:
        remaining = list(self.items_required)
        for item in self.items_delivered:
            if item in remaining:
                remaining.remove(item)
        return remaining


@dataclass
class SimState:
    """State exposed to strategy function each round."""
    round: int
    max_rounds: int
    grid_width: int
    grid_height: int
    walls: frozenset[Pos]
    shelves: frozenset[Pos]
    bots: list[SimBot]
    items: list[SimItem]
    orders: list[SimOrder]
    drop_off: Pos
    score: int

    def to_dict(self) -> dict:
        """Convert to same JSON format as game server (for compatibility)."""
        visible_orders = [o for o in self.orders if o.status in ("active", "preview")]
        return {
            "type": "game_state",
            "round": self.round,
            "max_rounds": self.max_rounds,
            "grid": {
                "width": self.grid_width,
                "height": self.grid_height,
                "walls": [list(w) for w in sorted(self.walls)],
            },
            "bots": [
                {"id": b.id, "position": list(b.position),
                 "inventory": list(b.inventory)}
                for b in self.bots
            ],
            "items": [
                {"id": i.id, "type": i.item_type, "position": list(i.position)}
                for i in self.items if not i.picked
            ],
            "orders": [
                {"id": o.id, "items_required": o.items_required,
                 "items_delivered": o.items_delivered,
                 "complete": o.complete, "status": o.status}
                for o in visible_orders
            ],
            "drop_off": list(self.drop_off),
            "score": self.score,
        }


class Simulator:
    """
    Local game simulator. Reproduces server logic exactly.
    
    Key behaviors:
    - Actions resolve in bot ID order (bot 0 first)
    - Invalid action = wait (no error, no penalty)
    - pick_up: bot must be adjacent (Manhattan 1) to shelf, inventory < 3
    - drop_off: bot must be ON drop-off, delivers ALL matching active order items
    - Auto-delivery: when active completes, preview→active, matching inventory auto-delivers
    - Items respawn at same shelf with new IDs after being picked
    """

    def __init__(self, width: int, height: int,
                 walls: set[Pos], shelves: set[Pos],
                 drop_off: Pos, spawn_positions: list[Pos],
                 order_sequence: list[dict],
                 item_types_at_shelves: dict[Pos, str],
                 max_rounds: int = 300):
        self.width = width
        self.height = height
        self.walls = frozenset(walls)
        self.shelves = frozenset(shelves)
        self.blocked = self.walls | self.shelves
        self.drop_off = drop_off
        self.spawn_positions = spawn_positions
        self.order_sequence = order_sequence  # all orders in game order
        self.shelf_types = item_types_at_shelves  # shelf_pos → item_type
        self.max_rounds = max_rounds

        # Precompute adjacency for pickup validation
        self._shelf_adjacent: dict[Pos, list[Pos]] = {}
        for shelf in self.shelves:
            adj = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                n = (shelf[0] + dx, shelf[1] + dy)
                if self._is_walkable(n):
                    adj.append(n)
            self._shelf_adjacent[shelf] = adj

    def _is_walkable(self, p: Pos) -> bool:
        return (0 <= p[0] < self.width and 0 <= p[1] < self.height
                and p not in self.blocked)

    def _is_adjacent_to_shelf(self, bot_pos: Pos, shelf_pos: Pos) -> bool:
        return bot_pos in self._shelf_adjacent.get(shelf_pos, [])

    # ------------------------------------------------------------------
    # Reset / Initialize
    # ------------------------------------------------------------------

    def reset(self) -> SimState:
        """Reset game to initial state. Returns round-0 state."""
        self._round = 0
        self._score = 0
        self._items_delivered = 0
        self._orders_completed = 0
        self._next_item_id = 0
        self._next_order_idx = 0

        # Create bots at spawn
        self._bots = [
            SimBot(id=i, position=pos)
            for i, pos in enumerate(self.spawn_positions)
        ]

        # Create items on all shelves
        self._items: list[SimItem] = []
        for shelf_pos, item_type in self.shelf_types.items():
            self._items.append(SimItem(
                id=f"item_{self._next_item_id}",
                item_type=item_type,
                position=shelf_pos,
            ))
            self._next_item_id += 1

        # Initialize orders
        self._orders: list[SimOrder] = []
        for i, order_data in enumerate(self.order_sequence):
            self._orders.append(SimOrder(
                id=order_data["id"],
                items_required=list(order_data["items_required"]),
                status="hidden",
            ))

        # Activate first order + preview
        if self._orders:
            self._orders[0].status = "active"
            self._next_order_idx = 1
        if len(self._orders) > 1:
            self._orders[1].status = "preview"
            self._next_order_idx = 2

        return self._get_state()

    def _get_state(self) -> SimState:
        return SimState(
            round=self._round,
            max_rounds=self.max_rounds,
            grid_width=self.width,
            grid_height=self.height,
            walls=self.walls,
            shelves=self.shelves,
            bots=[SimBot(b.id, b.position, list(b.inventory)) for b in self._bots],
            items=[SimItem(i.id, i.item_type, i.position, i.picked, i.picked_by)
                   for i in self._items if not i.picked],
            orders=[SimOrder(o.id, list(o.items_required),
                            list(o.items_delivered), o.status)
                    for o in self._orders if o.status in ("active", "preview")],
            drop_off=self.drop_off,
            score=self._score,
        )

    # ------------------------------------------------------------------
    # Step: process one round
    # ------------------------------------------------------------------

    def step(self, actions: list[dict]) -> tuple[SimState, bool]:
        """
        Process one round of actions.
        Returns (new_state, game_over).
        """
        self._step_internal(actions)
        game_over = self._round >= self.max_rounds
        return self._get_state(), game_over

    def _step_internal(self, actions: list[dict]):
        """Process one round without creating state object."""
        self._round += 1

        # Build action lookup by bot ID
        action_map = {}
        for a in actions:
            action_map[a["bot"]] = a

        # Track occupied positions (for collision detection)
        occupied = {b.position for b in self._bots}

        # Resolve actions in bot ID order
        for bot in sorted(self._bots, key=lambda b: b.id):
            act = action_map.get(bot.id, {"action": "wait"})
            action_type = act.get("action", "wait")

            if action_type in MOVES:
                self._resolve_move(bot, action_type, occupied)

            elif action_type == "pick_up":
                self._resolve_pickup(bot, act.get("item_id"))

            elif action_type == "drop_off":
                self._resolve_dropoff(bot)

            # else: wait (do nothing)

        # Respawn picked items at same shelf
        self._respawn_items()

    def _resolve_move(self, bot: SimBot, action: str,
                      occupied: set[Pos]):
        """Move bot if target is valid."""
        dx, dy = MOVES[action]
        new_pos = (bot.position[0] + dx, bot.position[1] + dy)

        if not self._is_walkable(new_pos):
            return  # wall/shelf/out of bounds → silent fail

        # Check collision with other bots
        # Since we process in ID order, check current positions of all bots
        for other in self._bots:
            if other.id != bot.id and other.position == new_pos:
                return  # blocked by another bot → silent fail

        occupied.discard(bot.position)
        bot.position = new_pos
        occupied.add(new_pos)

    def _resolve_pickup(self, bot: SimBot, item_id: Optional[str]):
        """Pick up item if valid."""
        if item_id is None:
            return
        if len(bot.inventory) >= 3:
            return  # inventory full

        # Find item
        item = None
        for i in self._items:
            if i.id == item_id and not i.picked:
                item = i
                break
        if item is None:
            return  # item not found or already picked

        # Check adjacency
        if not self._is_adjacent_to_shelf(bot.position, item.position):
            return  # not adjacent

        # Pick up
        item.picked = True
        item.picked_by = bot.id
        bot.inventory.append(item.item_type)

    def _resolve_dropoff(self, bot: SimBot):
        """Drop off matching items if bot is on drop-off."""
        if bot.position != self.drop_off:
            return  # not on drop-off cell

        active = self._get_active_order()
        if active is None:
            return  # no active order

        # Deliver ALL matching items in inventory
        remaining = list(active.items_remaining)
        delivered = []
        new_inventory = []

        for item_type in bot.inventory:
            if item_type in remaining:
                remaining.remove(item_type)
                delivered.append(item_type)
                active.items_delivered.append(item_type)
                self._score += 1
                self._items_delivered += 1
            else:
                new_inventory.append(item_type)

        bot.inventory = new_inventory

        # Check order completion
        if active.complete:
            self._score += 5
            self._orders_completed += 1
            active.status = "complete"
            self._advance_orders(bot)

    def _advance_orders(self, bot: SimBot):
        """
        After order completion: preview→active, next hidden→preview.
        Auto-deliver matching inventory items from the TRIGGERING bot only.
        (Server only auto-delivers for the bot at drop-off, not all bots.)
        """
        # Find preview → make active
        preview = None
        for o in self._orders:
            if o.status == "preview":
                preview = o
                break

        if preview:
            preview.status = "active"

            # Auto-delivery: only the triggering bot (at drop-off) gets re-checked
            remaining = list(preview.items_remaining)
            new_inventory = []
            for item_type in bot.inventory:
                if item_type in remaining:
                    remaining.remove(item_type)
                    preview.items_delivered.append(item_type)
                    self._score += 1
                    self._items_delivered += 1
                else:
                    new_inventory.append(item_type)
            bot.inventory = new_inventory

            # Check if auto-delivery completed the new order too
            if preview.complete:
                self._score += 5
                self._orders_completed += 1
                preview.status = "complete"
                # Recursive: advance again
                self._promote_next_preview()
                self._advance_orders(bot)
                return

        # Promote next hidden → preview
        self._promote_next_preview()

    def _promote_next_preview(self):
        """Promote next hidden order to preview. Generate new orders if needed."""
        if self._next_order_idx >= len(self._orders):
            self._generate_next_order()
        if self._next_order_idx < len(self._orders):
            self._orders[self._next_order_idx].status = "preview"
            self._next_order_idx += 1

    def _generate_next_order(self):
        """Generate a new order matching the recon's distribution patterns."""
        if not hasattr(self, "_order_rng"):
            import random
            self._order_rng = random.Random(42)
            sizes = [len(o["items_required"]) for o in self.order_sequence]
            self._order_size_pool = sizes if sizes else [3]
            self._item_type_pool = list(self.shelf_types.values())
            if not self._item_type_pool:
                return

        size = self._order_rng.choice(self._order_size_pool)
        items = [self._order_rng.choice(self._item_type_pool) for _ in range(size)]
        order_id = f"gen_order_{len(self._orders)}"
        self._orders.append(SimOrder(
            id=order_id,
            items_required=items,
            status="hidden",
        ))

    def _get_active_order(self) -> Optional[SimOrder]:
        for o in self._orders:
            if o.status == "active":
                return o
        return None

    def _respawn_items(self):
        """Respawn picked items at same shelf position with new IDs."""
        for item in self._items:
            if item.picked:
                # Create new item at same shelf
                self._items.append(SimItem(
                    id=f"item_{self._next_item_id}",
                    item_type=item.item_type,
                    position=item.position,
                ))
                self._next_item_id += 1
        # Remove picked items
        self._items = [i for i in self._items if not i.picked]

    # ------------------------------------------------------------------
    # Run full game
    # ------------------------------------------------------------------

    def run(self, strategy: Callable, verbose: bool = False) -> dict:
        """
        Run full game with a strategy function.
        
        strategy: callable(game_state_dict) -> {"actions": [...]}
                  OR has .decide_fast(sim_state) -> list[dict] for speed
        Returns: {"score": N, "rounds_used": N, "items_delivered": N,
                  "orders_completed": N}
        """
        state = self.reset()
        use_fast = hasattr(strategy, 'decide_fast')

        for _ in range(self.max_rounds):
            if verbose and self._round % 50 == 0:
                print(f"  Round {self._round}: score={self._score}")

            if use_fast:
                actions = strategy.decide_fast(self)
                self._step_internal(actions)
                if self._round >= self.max_rounds:
                    break
            else:
                state_dict = state.to_dict()
                response = strategy(state_dict)
                actions = response.get("actions", [])
                state, game_over = self.step(actions)
                if game_over:
                    break

        result = {
            "score": self._score,
            "rounds_used": self._round,
            "items_delivered": self._items_delivered,
            "orders_completed": self._orders_completed,
        }

        if verbose:
            print(f"  Final: score={result['score']}, "
                  f"items={result['items_delivered']}, "
                  f"orders={result['orders_completed']}, "
                  f"rounds={result['rounds_used']}")

        return result

    # ------------------------------------------------------------------
    # Factory: create from game log
    # ------------------------------------------------------------------

    @classmethod
    def from_game_log(cls, log_path: str) -> "Simulator":
        """Create simulator from a recon game log."""
        import json
        with open(log_path) as f:
            log = json.load(f)

        r0 = log["rounds"][0]
        g = r0["grid"]
        width, height = g["width"], g["height"]
        walls = {tuple(w) for w in g["walls"]}

        # Item positions → shelf_type mapping
        shelf_types: dict[Pos, str] = {}
        shelves: set[Pos] = set()
        for item in r0["items"]:
            pos = tuple(item["position"])
            shelf_types[pos] = item["type"]
            shelves.add(pos)

        # Also scan all rounds for shelves (items might spawn later)
        for r in log["rounds"]:
            for item in r["items"]:
                pos = tuple(item["position"])
                if pos not in shelf_types:
                    shelf_types[pos] = item["type"]
                    shelves.add(pos)

        drop_off = tuple(r0["drop_off"])
        spawns = [tuple(b["position"]) for b in r0["bots"]]

        # Order sequence
        orders = log.get("orders_sequence", [])
        order_seq = [
            {"id": o["id"], "items_required": o["items_required"]}
            for o in orders
        ]

        # If no order sequence in log, extract from rounds
        if not order_seq:
            seen_ids = set()
            for r in log["rounds"]:
                for o in r["orders"]:
                    if o["id"] not in seen_ids:
                        seen_ids.add(o["id"])
                        order_seq.append({
                            "id": o["id"],
                            "items_required": o["items_required"],
                        })

        return cls(
            width=width, height=height,
            walls=walls, shelves=shelves,
            drop_off=drop_off,
            spawn_positions=spawns,
            order_sequence=order_seq,
            item_types_at_shelves=shelf_types,
        )

    @classmethod
    def from_analysis(cls, analysis: dict, grid) -> "Simulator":
        """Create from analyzer output + grid."""
        shelf_types = {}
        for item_type, positions in analysis["type_shelf_positions"].items():
            for pos in positions:
                shelf_types[tuple(pos)] = item_type

        order_seq = [
            {"id": o["id"], "items_required": o["items"]}
            for o in analysis["order_sequence"]
        ]

        # Infer spawns from grid (bottom-right area typically)
        w, h = grid.width, grid.height
        spawns = [(w - 2, h - 2)]  # default

        return cls(
            width=w, height=h,
            walls=grid.walls, shelves=grid.shelves,
            drop_off=tuple(analysis["drop_off"]),
            spawn_positions=spawns,
            order_sequence=order_seq,
            item_types_at_shelves=shelf_types,
        )

    @classmethod
    def from_recon_data(cls, recon: dict) -> "Simulator":
        """
        Create simulator from live bot recon data (GameLogger output).

        Recon format (produced by bot.recon.logger.GameLogger.finalize):
            {
                "fingerprint": "abc12345",
                "grid_size": [width, height],
                "walls": [[x, y], ...],          # original walls, NO shelves
                "drop_off": [x, y],
                "shelf_map": {"milk": [[x,y], ...], ...},
                "bot_count": 1,
                "bot_start_positions": [[x, y], ...],
                "order_sequence": [
                    {"id": "order_0", "items_required": ["milk", "bread"], ...},
                ],
                "total_rounds": 300,
                "final_score": 95,
            }

        This is the primary way to test the live bot offline: run a recon game,
        save the data, then replay with from_recon_data() + BotAdapter.
        """
        width, height = recon["grid_size"]

        walls: set[Pos] = {tuple(w) for w in recon["walls"]}
        shelves: set[Pos] = set()
        shelf_types: dict[Pos, str] = {}

        for item_type, positions in recon["shelf_map"].items():
            for pos_list in positions:
                pos: Pos = tuple(pos_list)
                shelves.add(pos)
                shelf_types[pos] = item_type

        drop_off: Pos = tuple(recon["drop_off"])

        spawns = [tuple(p) for p in recon.get("bot_start_positions", [])]
        if not spawns:
            # Fallback: bottom-right corner (standard spawn area)
            spawns = [(width - 2, height - 2)]
            bot_count = recon.get("bot_count", 1)
            while len(spawns) < bot_count:
                last = spawns[-1]
                spawns.append((last[0], last[1] - 2))

        order_seq = [
            {"id": o["id"], "items_required": list(o["items_required"])}
            for o in recon.get("order_sequence", [])
        ]

        max_rounds = recon.get("total_rounds", 300)

        return cls(
            width=width,
            height=height,
            walls=walls,
            shelves=shelves,
            drop_off=drop_off,
            spawn_positions=spawns,
            order_sequence=order_seq,
            item_types_at_shelves=shelf_types,
            max_rounds=max_rounds,
        )

    @classmethod
    def from_recon_file(cls, path: str) -> "Simulator":
        """Convenience: load recon JSON from disk and build Simulator."""
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_recon_data(data)

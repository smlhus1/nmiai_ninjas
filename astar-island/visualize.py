"""Visualize Astar Island initial states and predictions."""

import json
import numpy as np
import sys

# Load round data
with open("data/astar_round1.json") as f:
    detail = json.load(f)

CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
CLASS_CHARS = {0: ".", 1: "S", 2: "P", 3: "R", 4: "F", 5: "M"}
CLASS_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

grid = detail["initial_states"][seed]["grid"]
settlements = detail["initial_states"][seed]["settlements"]

print(f"=== Seed {seed} — Initial State ===")
print(f"Settlements: {len(settlements)}")
print(f"Legend: . Empty  S Settlement  P Port  R Ruin  F Forest  M Mountain")
print()

# Count terrain types
counts = {}
for y in range(len(grid)):
    for x in range(len(grid[0])):
        cls = CODE_TO_CLASS.get(grid[y][x], 0)
        counts[cls] = counts.get(cls, 0) + 1

for cls, name in CLASS_NAMES.items():
    print(f"  {CLASS_CHARS[cls]} {name}: {counts.get(cls, 0)}")
print()

# Print grid
print("   " + "".join(f"{x % 10}" for x in range(40)))
for y in range(len(grid)):
    row = ""
    for x in range(len(grid[0])):
        cls = CODE_TO_CLASS.get(grid[y][x], 0)
        row += CLASS_CHARS[cls]
    print(f"{y:2d} {row}")

"""Watch for active Astar Island round with available query budget.

When a round is found with budget > 0, runs solver_v3 observation + prediction pipeline.
Designed to be called repeatedly (e.g. every 5 minutes).
"""

import sys
import requests

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEiLCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ.fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
BASE = "https://api.ainm.no"

s = requests.Session()
s.cookies.set("access_token", TOKEN)
s.headers["Origin"] = "https://app.ainm.no"

# Check for active round
rounds = s.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)

if not active:
    print("NO_ROUND: No active round.")
    sys.exit(0)

round_num = active["round_number"]
round_id = active["id"]
closes = active.get("closes_at", "?")
print(f"ACTIVE: R{round_num} ({round_id[:8]}) closes {closes}")

# Check budget
budget = s.get(f"{BASE}/astar-island/budget").json()
used = budget["queries_used"]
total = budget["queries_max"]
remaining = total - used

if remaining <= 0:
    print(f"BUDGET_SPENT: {used}/{total} queries used. Already observed.")
    sys.exit(0)

print(f"BUDGET_AVAILABLE: {remaining}/{total} queries remaining!")
print("READY — launching solver_v3...")
print("NOTIFY_DISCORD_WHEN_DONE")

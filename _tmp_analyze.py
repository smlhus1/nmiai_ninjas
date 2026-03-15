import json, sys
sys.path.insert(0, "C:\\Projects\\Personlig\\NmIAi")
from theoretical_max import load_recon, export_trips

recon = load_recon("C:\\Projects\\Personlig\\NmIAi\\logs/74001e7f_2026-03-13_recon.json")
data = export_trips(recon)
trips = data["trips"]
order_acts = data["order_activations"]

print(f"Total trips: {len(trips)}")
print(f"Total orders: {len(order_acts)}")
print(f"Bots: {data['n_bots']}")
print(f"Max rounds: {data['max_rounds']}")
print()

# Per-bot trip count
from collections import Counter
bot_trips = Counter()
for oi, t in trips:
    bot_trips[t.bot_id] += 1
print("Trips per bot:", dict(sorted(bot_trips.items())))
print()

# Trip timing details
print("Order | Bot | Start | Pickup_done | Delivery | Duration | Items | Dropoff")
for oi, t in trips[:30]:
    dur = t.delivery_round - t.start_round
    print(f"  {oi:3d}   {t.bot_id:3d}   {t.start_round:5d}   {t.pickup_done_round:11d}   {t.delivery_round:8d}   {dur:8d}   {len(t.items):5d}   {t.drop_off}")

print("\n... last 10:")
for oi, t in trips[-10:]:
    dur = t.delivery_round - t.start_round
    print(f"  {oi:3d}   {t.bot_id:3d}   {t.start_round:5d}   {t.pickup_done_round:11d}   {t.delivery_round:8d}   {dur:8d}   {len(t.items):5d}   {t.drop_off}")

# Active bots per round (how many bots are actively moving)
max_round = max(t.delivery_round for _, t in trips)
print(f"\nMax delivery round: {max_round}")

# Count concurrent trips per round
active_per_round = [0] * (max_round + 1)
for oi, t in trips:
    for r in range(t.start_round, min(t.delivery_round + 1, max_round + 1)):
        active_per_round[r] += 1

print(f"Peak concurrent trips: {max(active_per_round)}")
print(f"Avg concurrent trips: {sum(active_per_round)/len(active_per_round):.1f}")

# Show activity at key rounds
for r in [0, 10, 20, 50, 100, 150, 195]:
    if r < len(active_per_round):
        print(f"  Round {r}: {active_per_round[r]} active trips")

"""Quick E2E test: run BotAdapter through the Simulator."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Simulering.offline.test_e2e import make_easy_scenario
from Simulering.offline.bot_adapter import BotAdapter


def test_adapter_runs_easy_scenario():
    """Live bot (via BotAdapter) completes at least 1 order on easy scenario."""
    sim = make_easy_scenario()
    adapter = BotAdapter(suppress_logs=True)
    result = sim.run(adapter, verbose=False)
    recon = adapter.finalize(result)
    adapter.reset()

    assert result["score"] > 0, "Bot should score at least something"
    assert result["items_delivered"] > 0
    assert result["rounds_used"] > 0

    assert recon is not None
    assert "fingerprint" in recon
    assert len(recon["shelf_map"]) > 0
    assert len(recon["order_sequence"]) > 0


def test_adapter_recon_round_trip():
    """Recon data from adapter can recreate a working Simulator."""
    from Simulering.offline.simulator import Simulator

    sim1 = make_easy_scenario()
    adapter = BotAdapter(suppress_logs=True)
    result1 = sim1.run(adapter, verbose=False)
    recon = adapter.finalize(result1)
    adapter.reset()

    sim2 = Simulator.from_recon_data(recon)
    adapter2 = BotAdapter(suppress_logs=True)
    result2 = sim2.run(adapter2, verbose=False)
    adapter2.reset()

    assert result2["score"] >= 0
    assert result2["rounds_used"] > 0


def test_adapter_reset_works():
    """Adapter can be reused for multiple games after reset."""
    sim = make_easy_scenario()
    adapter = BotAdapter(suppress_logs=True)

    scores = []
    for _ in range(3):
        result = sim.run(adapter, verbose=False)
        scores.append(result["score"])
        adapter.finalize(result)
        adapter.reset()

    assert len(scores) == 3
    assert all(s >= 0 for s in scores)

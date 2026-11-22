import json
import math
from pathlib import Path

from t1_safety import ProtectedArena, StepTrace, horizontal

arena = ProtectedArena(19961107)
try:
    cfg = arena.prepare_episode(2026112104, 100)
    trace = StepTrace(Path("forced-death.csv.gz"))
    initial = horizontal(arena.info)
    with arena.track(trace):
        arena.command("/kill @p")  # Isolated negative control, outside the measured cohort.
        for step in range(1, 121):
            if not arena.navigation_step(arena.sim.noop_action()):
                break
    respawn = horizontal(arena.info)
    target = (arena.respawn_point[0]+.5, arena.respawn_point[2]+.5)
    assert trace.flags() == (1, True, 1) and trace.reason == "death"
    assert math.dist(respawn, target) <= 2 and math.dist(respawn, arena.landmark) >= 150
    result = dict(world_seed=arena.world_seed, seed=cfg["scenario_seed"], initial=initial,
                  actual_respawn=respawn, gold=arena.landmark,
                  respawn_distance=math.dist(respawn, arena.landmark), steps=step,
                  trace_flags=trace.flags(), status=trace.reason, health_after=trace.rows[-1][6],
                  **arena.safety_config())
    Path("summary.json").write_text(json.dumps(result, indent=2, default=float)+"\n")
    print(json.dumps(result, default=float), flush=True)
finally:
    arena.sim.close()

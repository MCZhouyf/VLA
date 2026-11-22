import json
import math
import os
from pathlib import Path

from t1_v4 import RandomHeadingArena, horizontal, wrapped_angle

root = Path(os.environ['WORK']) / 'results/t1-v4-acceptance-K6ARCMSV'
path, = root.glob('worker-*/d100-seed2026112143.json')
fixture = json.loads(path.read_text())
arena = RandomHeadingArena(fixture['world_seed'])
try:
    arena.prepare_episode(fixture['scenario_seed'], 100)
    arena.restore_return_pose(fixture)
    steps = arena.turn_to(wrapped_angle(arena.info["player_pos"]["yaw"] + 180))
    action = arena.sim.noop_action()
    action["forward"] = 1
    initial = horizontal(arena.info)
    print("initial", initial, "landmark", arena.landmark, flush=True)
    minimum = math.dist(initial, arena.landmark)
    for step in range(steps + 1, fixture["return_limit"] + 1):
        before = arena.info['player_pos'].copy()
        alive_before = arena.info.get('life_stats')
        if not arena.navigation_step(action.copy()):
            print("rejected_step", step, arena.navigation_end, "closest", minimum, flush=True)
            print("before", before, "life", alive_before, flush=True)
            print("after", arena.info['player_pos'], "life", arena.info.get('life_stats'), flush=True)
            break
        minimum = min(minimum, math.dist(horizontal(arena.info), arena.landmark))
    else:
        print("budget reached", "closest", minimum, flush=True)
finally:
    arena.sim.close()

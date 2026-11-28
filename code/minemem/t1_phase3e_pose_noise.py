"""Apply matched per-step measurement noise to the frozen spatial planner."""

import math
import random
from pathlib import Path

from t1_phase3e_predicted_rollout import GridPlanPolicy
from t1_respawn import record_config


LEVELS = ((0.05, 0.5), (0.1, 1.0), (0.2, 2.0))


class PoseNoise:
    def __init__(self, seed, position_sigma, yaw_sigma):
        self.rng = random.Random(f"t1-pose-measurement:{seed}")
        self.position_sigma = position_sigma
        self.yaw_sigma = yaw_sigma
        self.n = 0
        self.position_squared = 0.0
        self.yaw_squared = 0.0

    def observe(self, pose):
        dx, dz = [self.rng.gauss(0, self.position_sigma) for _ in range(2)]
        dyaw = self.rng.gauss(0, self.yaw_sigma)
        self.n += 1
        self.position_squared += dx**2 + dz**2
        self.yaw_squared += dyaw**2
        return pose[0]+dx, pose[1]+dz, (pose[2]+dyaw+180) % 360-180

    def report(self):
        return dict(samples=self.n, horizontal_rmse=math.sqrt(self.position_squared/self.n),
                    yaw_rmse=math.sqrt(self.yaw_squared/self.n))


def install_noise(evaluator, output, position_sigma, yaw_sigma):
    if (position_sigma, yaw_sigma) not in LEVELS:
        raise ValueError("Choose one of the three registered pose noise levels")
    policies = {}

    class NoisyPlanPolicy(GridPlanPolicy):
        def __init__(self, models, outbound, seed, shuffle_visual, tracking_repair=False):
            self.noise = PoseNoise(seed, position_sigma, yaw_sigma)
            measured = [(image, self.noise.observe(pose), *rest) for image, pose, *rest in outbound]
            super().__init__(models, measured, seed, shuffle_visual, tracking_repair)
            policies[seed] = self

        def __call__(self, info):
            player = info["player_pos"]
            pose = self.noise.observe([float(player[key]) for key in ("x", "z", "yaw")])
            return super().__call__(dict(info, player_pos=dict(player, **dict(zip(("x","z","yaw"), pose)))))

    run_seed = evaluator.run_seed

    def score_lane(args, cases, controller, models, emit):
        def logged(row):
            distance, seed = row[:2]
            noise = policies.pop(seed).noise
            record_config(Path(output), f"d{distance}-seed{seed}-pose-noise", noise.report())
            emit(row)

        return run_seed(args, cases, controller, models, logged)

    evaluator.GridPlanPolicy = NoisyPlanPolicy
    evaluator.run_seed = score_lane

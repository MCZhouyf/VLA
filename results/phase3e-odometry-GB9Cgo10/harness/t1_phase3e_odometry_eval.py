"""Use executed actions as the only source of the planner's live pose."""

import json
import math
from pathlib import Path

from t1_phase3e_odometry import ActionIntegrator
from t1_phase3e_predicted_rollout import GridPlanPolicy
from t1_phase3e_repaired_scene import GroundRepairedArena
from t1_respawn import record_config
from t1_trained_eval import ControlTrace


def install_odometry(evaluator, output, calibration, ground_repair_world=None):
    parameters = json.loads(Path(calibration).read_text())
    integration_keys = ("drag","acceleration")
    settings = {key:parameters[key] for key in integration_keys}
    policies = {}

    class ActionPoseArena:
        mapping = False
        integrator = None

        def estimate(self):
            state = self.integrator
            return float(state.position[0]),float(state.position[1]),state.yaw

        def outward(self, layout, trace):
            self.integrator = ActionIntegrator(**settings)
            self.mapping = True
            try:
                return super().outward(layout,trace)
            finally:
                self.mapping = False

        def teacher_step(self, yaw_delta=0., forward=False, jump=False, record=False):
            writing = self.mapping and self.capture is not None and record
            before = self.estimate() if writing else None
            super().teacher_step(yaw_delta,forward,jump,record)
            if writing:
                image,_,*actions = self.capture[-1]
                self.capture[-1] = (image,before,*actions)

        def navigation_step(self, action):
            valid = super().navigation_step(action)
            if self.mapping or isinstance(self.trace,ControlTrace):
                self.info["estimated_pose"] = self.integrator.step(float(action["camera"][1]),action)
            return valid

        def restore_return_pose(self, config):
            super().restore_return_pose(config)
            if self.integrator is not None:
                self.info["estimated_pose"] = self.estimate()

    class OriginalArena(ActionPoseArena,evaluator.FirstArrivalArena):
        pass

    class RepairedArena(ActionPoseArena,GroundRepairedArena):
        pass

    def make_arena(world_seed, effects):
        kind = RepairedArena if world_seed==ground_repair_world else OriginalArena
        return kind(world_seed,effects)

    class OdometryPlanPolicy(GridPlanPolicy):
        def __init__(self, models, outbound, seed, shuffle_visual, tracking_repair=False):
            super().__init__(models,outbound,seed,shuffle_visual,tracking_repair)
            policies[seed] = self
            self.alignment = None
            self.samples = 0
            self.position_squared = self.yaw_squared = 0.

        def __call__(self, info):
            estimate = info["estimated_pose"]
            actual = info["player_pos"]
            # Scoring aligns the two frames at the start of return. The planner receives only estimate.
            if self.alignment is None:
                self.alignment = (estimate,(actual["x"],actual["z"],actual["yaw"]))
            initial, origin = self.alignment
            angle = math.radians(origin[2]-initial[2])
            dx,dz = estimate[0]-initial[0],estimate[1]-initial[1]
            x = origin[0]+dx*math.cos(angle)-dz*math.sin(angle)
            z = origin[1]+dx*math.sin(angle)+dz*math.cos(angle)
            yaw = (estimate[2]+origin[2]-initial[2]-actual["yaw"]+180)%360-180
            self.samples += 1
            self.position_squared += (x-actual["x"])**2+(z-actual["z"])**2
            self.yaw_squared += yaw*yaw
            return super().__call__({"player_pos":dict(zip(("x","z","yaw"),estimate))})

        def report(self):
            return dict(samples=self.samples,horizontal_rmse=math.sqrt(self.position_squared/self.samples),
                        yaw_rmse=math.sqrt(self.yaw_squared/self.samples))

    run_seed = evaluator.run_seed

    def score_lane(args, cases, controller, models, emit):
        if controller.policy.memory is not None:
            raise ValueError("Action-only rollouts require the shared pose-free controller")
        def logged(row):
            distance, seed = row[:2]
            record_config(Path(output),f"d{distance}-seed{seed}-odometry",policies.pop(seed).report())
            emit(row)

        return run_seed(args,cases,controller,models,logged)

    evaluator.FirstArrivalArena = make_arena
    evaluator.GridPlanPolicy = OdometryPlanPolicy
    evaluator.run_seed = score_lane

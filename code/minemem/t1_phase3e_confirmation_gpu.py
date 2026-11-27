"""Run the three confirmation ablations and material-swap pairs with frozen components."""

import argparse
import csv
import hashlib
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from t1_dual_enclosure import configure_rendering
from t1_phase3e import FirstArrivalArena
from t1_phase3e_closed_audit import large_turn_switches
from t1_phase3e_eval import FIELDS, checked_scene, movement_summary
from t1_phase3e_grid_plan import heading_turn
from t1_phase3e_no_anchor import STRIDE
from t1_phase3e_pairs import PairedArena, encode_view
from t1_phase3e_predicted_rollout import GridPlanPolicy, Models
from t1_respawn import record_config
from t1_safety import StepTrace
from t1_trained_eval import ControlTrace, Controller, drive_rollouts, rollout
from t1_waypoint import RouteLookahead, WaypointTurnStabilizer, waypoint_instruction, waypoint_labels


MODES = ("b1", "b2", "b3", "twin")
EXTRA = ("arm", "material_slot", "predicted_slot", "goal_slot", "stabilizer_steps",
         "heading_guard_steps", "waypoint_jumps_tail200")


def pose(info):
    return tuple(float(info["player_pos"][key]) for key in ("x", "z", "yaw"))


class OraclePolicy:
    def __init__(self, config):
        start=config["return_pose"]
        self.route=RouteLookahead((start["x"],start["z"]),config["teacher_targets"])
        self.slot=config["goal_slot"]

    def __call__(self, info):
        current=pose(info)
        target=self.route.target(current)
        self.turn=heading_turn(current,target)
        return waypoint_instruction(*waypoint_labels(current,target))

    def corrective_yaw(self):
        return self.turn


class ScriptController:
    policy = SimpleNamespace(history=2)

    def start(self, _):
        pass


def capture_outbound(arena, config, path):
    arena.capture=[]
    try:
        arena.outward(config,StepTrace(path))
        return arena.capture
    finally:
        arena.capture=None
        arena.distractor=None


def drive_script(generator, arena, trace):
    # The frozen heading guard supplies the turn on off-axis steps.
    forward=arena.sim.noop_action()
    forward["forward"]=1
    agent=arena.sim.env_action_to_agent_action({key:np.asarray([value]) for key,value in forward.items()})
    agent={key:int(np.asarray(value).item()) for key,value in agent.items()}
    try:
        next(generator)
        while True:
            trace.record_prediction([],agent,"script_forward",True,[])
            generator.send((agent,True))
    except StopIteration as done:
        return done.value
    finally:
        generator.close()


def play(args, arena, controller, config, outbound, policy, tag, arm, emit):
    arena.landmark,arena.distractor=config["landmark"],config["distractor"]
    record_config(args.output,tag,config)
    trace=ControlTrace(args.output/f"{tag}-return.csv.gz",args.output/f"{tag}-actions.jsonl")
    budget=config["return_limit"]
    episode=rollout(arena,controller,config,outbound[-controller.policy.history:],policy,budget,trace,
                    stabilize_turns=None if args.mode=="b2" else WaypointTurnStabilizer(),
                    heading_correction=None if args.mode=="b1" else policy.corrective_yaw)
    if args.mode=="b2":
        score=drive_script(episode,arena,trace)
    else:
        score=yield from episode
    overrides=[item.get("control_override") for item in trace.controls]
    emit((30,config["scenario_seed"],config["world_seed"],*score,budget,*movement_summary(trace),
          arm,config["goal_slot"],policy.slot,config["goal_slot"],
          overrides.count("waypoint_turn_stabilizer"),overrides.count("route_heading_guard"),
          large_turn_switches(trace.controls)))
    arena.distractor=None


def ablation(args, model, controller, config, emit):
    arena=FirstArrivalArena(config["world_seed"],config["optional_effects"])
    try:
        tag=f"d30-seed{config['scenario_seed']}"
        outbound=capture_outbound(arena,checked_scene(arena,config),args.output/f"{tag}-outbound.csv.gz")
        checked_scene(arena,config)
        if args.mode=="b3":
            policy=OraclePolicy(config)
        else:
            policy=GridPlanPolicy(model,outbound,config["scenario_seed"],False,True)
        yield from play(args,arena,controller,config,outbound,policy,tag,args.mode,emit)
    finally:
        arena.sim.close()


def twin(args, model, controller, config, emit):
    seed=config["scenario_seed"]
    arena=PairedArena(config["world_seed"],config["optional_effects"])
    try:
        layout=checked_scene(arena,config)
        canonical,_,_=arena.track_route(layout,args.output/f"d30-seed{seed}-outbound-error.csv.gz")
        canonical=canonical[:-1]
        indices=sorted(set(range(0,len(canonical),STRIDE)) |
                       set(range(len(canonical)-controller.policy.history,len(canonical))))
        recorded={}
        for slot in (config["goal_slot"],1-config["goal_slot"]):
            arena.material_slot(config,slot)
            outbound=[(None,[x,z,yaw],controller.policy.actions.null_token()) for x,_,z,yaw,_ in canonical]
            sampled=[]
            image_hashes=[]
            for index in indices:
                actual,picture=arena.render_pose(canonical[index])
                image=encode_view(picture)
                sampled.append(actual)
                image_hashes.append(hashlib.sha256(image).hexdigest())
                outbound[index]=(image,outbound[index][1],outbound[index][2])
            recorded[slot]=(outbound,sampled,image_hashes)
        if recorded[0][1]!=recorded[1][1]:
            raise RuntimeError(f"Material swap changed the observation poses for seed {seed}")
        record_config(args.output,f"d30-seed{seed}-pair",dict(seed=seed,canonical=canonical,sampled_indices=indices,
                     matched_render_poses=recorded[0][1],image_sha256={i:recorded[i][2] for i in (0,1)}))
        for slot in (config["goal_slot"],1-config["goal_slot"]):
            swapped=dict(config,goal_slot=slot,landmark=config["candidate_positions"][slot],
                         distractor=config["candidate_positions"][1-slot])
            for arm in ("visual","shuffle_visual"):
                arena.material_slot(config,slot)
                outbound=recorded[slot][0]
                tag=f"d30-seed{seed}-slot{slot}-{arm}"
                policy=GridPlanPolicy(model,outbound,seed,arm=="shuffle_visual",True)
                yield from play(args,arena,controller,swapped,outbound,policy,tag,arm,emit)
    finally:
        arena.sim.close()


def run(args):
    fixtures=json.loads(args.fixtures.read_text())
    cases=[fixtures["fixtures"][f"30/{seed}"] for seed in args.seeds]
    if len({case["scenario_seed"] for case in cases})!=len(cases):
        raise ValueError("Duplicate frozen ablation or pair seed")
    args.variant="spatial"
    args.grid_plan="visual"
    model=None if args.mode=="b3" else Models(args)
    controller=ScriptController() if args.mode=="b2" else Controller(args.checkpoint,args.training_run)
    configure_rendering(args.output)
    with (args.output/"episodes.csv").open("x",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow((*FIELDS,*EXTRA))
        def emit(row):
            writer.writerow(row)
            stream.flush()
        target=twin if args.mode=="twin" else ablation
        for index in range(0,len(cases),args.batch):
            stage=cases[index:index+args.batch]
            jobs=[target(args,model,copy(controller),config,emit) for config in stage]
            drive_rollouts(jobs,lambda *_:None)
            print(f"done={min(index+args.batch,len(cases))}/{len(cases)}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("output","checkpoint","training-run","fixtures","room-weights"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--mode",choices=MODES,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    parser.add_argument("--batch",type=int,default=3)
    run(parser.parse_args())

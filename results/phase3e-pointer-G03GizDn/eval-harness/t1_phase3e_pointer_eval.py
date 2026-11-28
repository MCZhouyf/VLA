"""Evaluate learned cell pointers and the matched shortest-path control."""

import argparse
import hashlib
import math
from pathlib import Path

import torch

import t1_phase3e_confirmation_gpu as experiment
from t1_phase3e_grid_plan import CommittedGridRoute, heading_turn
from t1_phase3e_no_anchor import STRIDE
from t1_phase3e_pointer import CellMemory, CellPointers, PointerPolicy
from t1_phase3e_predicted_rollout import open_jpeg
from t1_phase3e_repaired_scene import GroundRepairedArena
from t1_waypoint import waypoint_instruction, waypoint_labels


class PointerPairedArena(GroundRepairedArena,experiment.PairedArena):
    pass


class LearnedGoalGrid(PointerPolicy):
    def __init__(self, model, memory, outbound):
        super().__init__(model,memory)
        self.route = CommittedGridRoute(outbound,goal=self.goal_point)

    def __call__(self, info):
        pose = experiment.pose(info)
        target = self.route.target(pose)
        self.turn = heading_turn(pose,target)
        return waypoint_instruction(*waypoint_labels(pose,target))


def run(args):
    checkpoint = torch.load(args.pointer_weights,map_location="cpu",weights_only=True)
    if hashlib.sha256(args.room_weights.read_bytes()).hexdigest()!=checkpoint["room_weights_sha256"]:
        raise ValueError("Use the frozen visual projection from training")
    pointer = CellPointers(checkpoint["visual_width"]).to("cuda").eval()
    pointer.load_state_dict(checkpoint["state"])

    @torch.inference_mode()
    def create_policy(models, outbound, seed, shuffle_visual, tracking_repair=False):
        indices = sorted(set(range(0,len(outbound),STRIDE)) | {len(outbound)-1})
        poses = [frame[1] for frame in outbound]
        frames = models.images([open_jpeg(outbound[i][0]) for i in indices],(models.side,))[models.side]
        appearance = models.rooms.embed(frames)
        if shuffle_visual:
            permutation = torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed))
            appearance = appearance[permutation.to(appearance.device)]
        memory = CellMemory(poses,[poses[i] for i in indices],appearance)
        policy = (LearnedGoalGrid(pointer,memory,poses) if args.pointer_type=="grid"
                  else PointerPolicy(pointer,memory))
        policy.slot = -1
        return policy

    play = experiment.play

    def scored_play(args, arena, config, outbound, policy, tag, arm, emit):
        def score_row(row):
            # Room identity is a score only. The policy has already finished the episode.
            candidates = config["candidate_positions"]
            room = min(range(len(candidates)),key=lambda slot: math.dist(policy.goal_point,candidates[slot]))
            index = len(experiment.FIELDS)+2
            emit((*row[:index],room,*row[index+1:]))

        return play(args,arena,config,outbound,policy,tag,arm,score_row)

    experiment.GridPlanPolicy = create_policy
    experiment.play = scored_play
    experiment.FirstArrivalArena = GroundRepairedArena
    experiment.PairedArena = PointerPairedArena
    experiment.run(args)


def main(arguments, twin):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output","checkpoint","training-run","fixtures","room-weights","pointer-weights"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--pointer-type",choices=("learned","grid"),required=True)
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    parser.add_argument("--batch",type=int,default=3)
    args = parser.parse_args(arguments)
    args.mode = "twin" if twin else "pointer"
    args.repaired_scene = False
    run(args)

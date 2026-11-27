"""Run learned and grid-planned waypoints with the shared Phase A controller."""

import csv
import hashlib
import io
import json
from collections import deque
from copy import copy
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from t1_dual_enclosure import configure_rendering
from t1_memory_cache import encode_image_batch
from t1_phase3e import PROTOCOL, FirstArrivalArena
from t1_phase3e_closed_audit import large_turn_switches
from t1_phase3e_eval import FIELDS, checked_scene, movement_summary
from t1_phase3e_grid_plan import TRACKING_REPAIR, CommittedGridRoute, GridRoute, heading_turn
from t1_phase3e_no_anchor import AnchorFreeReadout, STRIDE
from t1_phase3e_pairs import encode_view
from t1_phase3e_pair_features import FrozenVision, RESOLUTION, TAIL
from t1_phase3e_pair_probe import RoomReadout, history_plan, window_plan
from t1_phase3e_waypoint_heads import Predictor, short_plan
from t1_respawn import record_config
from t1_safety import StepTrace
from t1_spatial import spatial_read_plan
from t1_trained_eval import ControlTrace, Controller, drive_rollouts, rollout
from t1_waypoint import WaypointTurnStabilizer, waypoint_instruction, waypoint_labels


def tensors(parts):
    return tuple(torch.as_tensor(part,device="cuda") for part in parts)


class Models:
    def __init__(self,args):
        self.variant = args.variant
        checkpoint = torch.load(args.room_weights,map_location="cpu",weights_only=True)
        self.side = checkpoint["specification"]["patch_side"]
        self.rooms = RoomReadout(checkpoint["specification"])
        if args.grid_plan:
            self.rooms.load_state_dict(checkpoint["room_state"])
            self.rooms.to("cuda").eval()
            self.attention = AnchorFreeReadout(self.rooms.reader.width-32)
            self.attention.load_state_dict(checkpoint["attention_state"])
            self.attention.to("cuda").eval()
            self.vision = FrozenVision(args.checkpoint,2,12)
            return
        learned = torch.load(args.waypoint_weights,map_location="cpu",weights_only=True)
        if not learned.get("candidate_conditioned",False):
            raise ValueError("The waypoint predictor must use its own predicted candidate")
        initial = torch.load(args.initial,map_location="cpu",weights_only=True)
        self.waypoints = Predictor(initial,condition_candidate=True)
        self.waypoints.load_state_dict(learned["models"][args.variant])
        self.waypoints.to("cuda").eval()
        self.room_variant = "current" if args.variant=="no_memory" else args.variant
        self.rooms.load_state_dict(checkpoint["models"][self.room_variant])
        self.rooms.to("cuda").eval()
        self.vision = FrozenVision(args.checkpoint,2,12)

    @torch.inference_mode()
    def images(self, pictures, sides):
        if any(picture.size!=RESOLUTION for picture in pictures):
            pictures = [picture.resize(RESOLUTION,Image.Resampling.BILINEAR) for picture in pictures]
        return {side:torch.stack([grid for start in range(0,len(pictures),self.vision.batch)
                                  for grid,_ in encode_image_batch(self.vision.encoder,self.vision.processor,
                                    pictures[start:start+self.vision.batch],side,"cuda")]).to("cuda")
                for side in sides}


def open_jpeg(image):
    with Image.open(io.BytesIO(image)) as source:
        return source.convert("RGB")


class WaypointPolicy:
    def __init__(self,models,outbound,anchors):
        self.models = models
        self.variant = models.variant
        self.slot = None
        self.sides = {2,models.side}
        self.observe_full = self.variant=="spatial"
        self.tail = {side:deque(maxlen=TAIL+1) for side in self.sides}
        images = models.images([open_jpeg(frame[0]) for frame in outbound[-TAIL:]],self.sides)
        for side,values in images.items():
            self.tail[side].extend(values)
        if self.observe_full:
            indices = sorted(set(range(0,len(outbound),8)) | {len(outbound)-1})
            self.history = models.images([open_jpeg(outbound[i][0]) for i in indices],(2,))[2]
            self.poses = [outbound[i][1] for i in indices]
            room_indices = sorted(set(range(0,len(outbound)-TAIL,8)) | set(anchors))
            self.room_poses = [outbound[i][1] for i in room_indices]
            self.anchors = [room_indices.index(i) for i in anchors]
            self.room_grid = models.images([open_jpeg(outbound[i][0]) for i in room_indices],(models.side,))[models.side]

    def choose_room(self,pose):
        window = torch.stack(list(self.tail[self.models.side]))
        if self.observe_full:
            memory = self.room_grid
            plan = history_plan(self.room_poses,[pose],self.models.rooms.reader.grid_side,self.anchors)
        else:
            memory = window
            plan = window_plan(0 if self.variant=="no_memory" else TAIL,states=1)
        label,_,_ = self.models.rooms(dict(shared=window,history=memory,
                                           plans={self.models.room_variant:tensors(plan)}),self.models.room_variant)
        return int(label.argmax(-1).item())

    @torch.inference_mode()
    def __call__(self,info):
        if self.slot is None:
            active = self.sides
        else:
            active = (2,)
        pictures = self.models.images([open_jpeg(encode_view(info["pov"]))],active)
        for side,value in pictures.items():
            self.tail[side].append(value[0])
        if self.observe_full:
            pose = [float(info["player_pos"][key]) for key in ("x","z","yaw")]
        else:
            pose = None
        if self.slot is None:
            self.slot = self.choose_room(pose)
        if self.observe_full:
            grid = torch.cat((self.history,self.tail[2][-1][None]),dim=0)
            index = len(grid)-1
            read = tuple(field[None] for field in spatial_read_plan(self.poses,pose,
                                                      self.models.waypoints.reader.grid_side))
        else:
            grid = torch.stack(list(self.tail[2]))
            index = len(grid)-1
            read = short_plan([index],dict(enumerate(range(len(grid)))),2 if self.variant=="no_memory" else TAIL)
        case = dict(grid=grid,current=torch.as_tensor([index],device=grid.device),probes={},
                    plans={self.variant:tensors(read)},candidate=self.slot)
        direction,distance,*_ = self.models.waypoints(case,self.variant)
        return waypoint_instruction(int(direction.argmax(-1).item()),int(distance.argmax(-1).item()))


class GridPlanPolicy:
    @torch.inference_mode()
    def __init__(self, models, outbound, seed, shuffle_visual, tracking_repair=False):
        sampled = outbound[::STRIDE]
        images = models.images([open_jpeg(frame[0]) for frame in sampled],(models.side,))[models.side]
        appearance = models.rooms.embed(images)
        poses = torch.tensor([frame[1] for frame in sampled],device=appearance.device)
        if shuffle_visual:
            permutation = torch.randperm(len(sampled),generator=torch.Generator().manual_seed(seed))
            appearance = appearance[permutation.to(appearance.device)]
        self.slot = models.attention(appearance,poses).argmax().item()
        route_type = CommittedGridRoute if tracking_repair else GridRoute
        self.route = route_type([frame[1] for frame in outbound], self.slot)
        self.track = tracking_repair
        self.turn = None

    def __call__(self, info):
        pose = tuple(float(info["player_pos"][key]) for key in ("x","z","yaw"))
        target = self.route.target(pose)
        if self.track:
            self.turn = heading_turn(pose, target)
        return waypoint_instruction(*waypoint_labels(pose, target))

    def corrective_yaw(self):
        return self.turn


def run_seed(args,cases,controller,models,emit):
    worlds = {(case["world_seed"],case["optional_effects"]) for case in cases}
    if len(worlds)!=1:
        raise ValueError("A rollout lane must use a single world")
    arena = FirstArrivalArena(*worlds.pop())
    try:
        staged=[]
        for expected in cases:
            config=checked_scene(arena,expected)
            tag=f"d{expected['distance']}-seed{expected['scenario_seed']}"
            arena.capture=[]
            try:
                arena.outward(config,StepTrace(args.output/f"{tag}-outbound.csv.gz"))
                anchors = None if args.grid_plan else config["room_frame_indices"]
                staged.append((expected,tag,arena.capture,anchors))
            finally:
                arena.capture=None
                arena.distractor=None
        for expected,tag,outbound,anchors in staged:
            checked_scene(arena,expected)
            arena.landmark,arena.distractor=expected["landmark"],expected["distractor"]
            record_config(args.output,tag,expected)
            if args.grid_plan:
                policy=GridPlanPolicy(models,outbound,expected["scenario_seed"],
                                      args.grid_plan=="shuffle_visual",args.tracking_repair)
            else:
                policy=WaypointPolicy(models,outbound,anchors)
            trace=ControlTrace(args.output/f"{tag}-return.csv.gz",args.output/f"{tag}-actions.jsonl")
            limit=expected["return_limit"]
            result=yield from rollout(arena,controller,expected,outbound[-controller.policy.history:],policy,
                                      limit,trace,stabilize_turns=None if args.no_stabilizer else WaypointTurnStabilizer(),
                                      heading_correction=policy.corrective_yaw if args.tracking_repair else None)
            interventions=sum(row.get("control_override")=="waypoint_turn_stabilizer" for row in trace.controls)
            row = (expected["distance"],expected["scenario_seed"],expected["world_seed"],*result,limit,
                   *movement_summary(trace),policy.slot,expected["goal_slot"],interventions)
            if args.grid_plan:
                row += (large_turn_switches(trace.controls),)
            if args.tracking_repair:
                row += (sum(item.get("control_override")=="route_heading_guard" for item in trace.controls),)
            emit(row)
            arena.distractor=None
    finally:
        arena.sim.close()


def run(args):
    gate=json.loads(args.offline_gate.read_text())
    if not gate["passed"] or gate["weights_sha256"]!=hashlib.sha256(args.room_weights.read_bytes()).hexdigest():
        raise ValueError("Use the exact readout that passed the paired offline gate")
    if not json.loads(args.oracle_gate.read_text())["passed"]:
        raise ValueError("The identical Phase A control protocol must pass its oracle gate")
    metrics=json.loads(args.metrics.read_text())
    fixtures=json.loads(args.fixtures.read_text())
    if not metrics["passed"] or fixtures["protocol"]!=PROTOCOL:
        raise ValueError("Use the accepted T1 v3 layout")
    selected=args.seeds or fixtures["formal"]
    if len(selected)!=len(set(selected)) or not set(selected)<=set(fixtures["formal"]):
        raise ValueError("Select distinct frozen seeds")
    cases=[[fixtures["fixtures"][f"{d}/{s}"] for d in args.distances] for s in selected]
    controller=Controller(args.checkpoint,args.training_run)
    if controller.settings.get("evaluation_history")!="placeholder":
        raise ValueError("The frozen waypoint controller must use placeholder action history")
    models=Models(args)
    configure_rendering(args.output)
    protocol = dict(stabilize_turns=not args.no_stabilizer,variant=args.variant,
                    distances=args.distances,seeds=selected)
    if args.grid_plan:
        protocol.update(grid_plan=args.grid_plan, room_anchor_input=False)
    if args.tracking_repair:
        protocol["tracking_repair"] = TRACKING_REPAIR
    record_config(args.output,"control_protocol",protocol)
    with (args.output/"episodes.csv").open("x",newline="") as stream:
        writer=csv.writer(stream)
        fields = (*FIELDS,"predicted_slot","goal_slot","stabilizer_steps")
        if args.grid_plan:
            fields += ("waypoint_jumps_tail200",)
        if args.tracking_repair:
            fields += ("heading_guard_steps",)
        writer.writerow(fields)
        for offset in range(0,len(cases),args.batch):
            stage=cases[offset:offset+args.batch]
            start=perf_counter()
            def emit(row):
                writer.writerow(row)
                stream.flush()
            drive_rollouts([run_seed(args,lane,copy(controller),models,emit) for lane in stage],lambda *_:None)
            print(f"done={min(offset+args.batch,len(cases))}/{len(cases)} seconds={perf_counter()-start:.1f}",flush=True)

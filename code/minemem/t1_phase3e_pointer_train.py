"""Fit and audit Plan L on existing T1 v3 visual caches."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phase1_metrics import wilson
from t1_phase3e_no_anchor import metadata, STRIDE
from t1_phase3e_pair_probe import RoomReadout, TAIL, load_cache
from t1_phase3e_pointer import CellMemory, CellPointers, nearest_cells
from t1_waypoint import RouteLookahead, teacher_waypoints


def project(encoder, grid):
    with torch.no_grad():
        return encoder.embed(grid)


def case(memory, scene, slot, poses, waypoints, row):
    candidates = torch.tensor(scene["candidate_positions"],dtype=memory.points.dtype)
    return dict(memory=memory,row=row,slot=slot,candidates=candidates,
                goal=int(nearest_cells(memory.points,candidates[slot])[0]),
                pose=memory.local_pose(poses),waypoints=nearest_cells(memory.points,waypoints))


def training(plan, cache, encoder):
    for row in json.loads(plan.read_text())["episodes"]:
        if row["role"] not in ("train","dart") or not isinstance(row["group"],int):
            continue
        record = load_cache(cache,row["scene_seed"])
        if record["row"]!=row:
            raise ValueError(f"Training cache identity changed for scene {row['scene_seed']}")
        meta = record["meta"]
        start = meta["outbound_frames"]
        poses = np.asarray(meta["pose"])
        sampled = record["history"]
        stored = {frame:index for index,frame in enumerate(record["frame_indices"])}
        appearance = project(encoder,record["grid"][[stored[index] for index in sampled]])
        memory = CellMemory(poses[:start],poses[sampled],appearance)
        end = start+row["selected_labels"]
        scene = meta["scene"]
        identity = dict(seed=row["scene_seed"],world=row["world_seed"],distance=row["group"],split=row["role"])
        yield case(memory,scene,scene["goal_slot"],poses[start:end],
                   teacher_waypoints(meta)[:end-start],identity)


def heldout(index, cache, encoder):
    for row in json.loads(index.read_text())["archives"]:
        record = load_cache(cache,row["seed"])
        if any(record[k]!=row[k] for k in ("split","distance","seed","world")):
            raise ValueError(f"Held-out cache identity changed for scene {row['seed']}")
        meta = metadata(row["path"])
        poses = np.asarray(meta["canonical_outbound_pose"][:-1])[:,[0,2,3]]
        start = len(poses)
        history = meta["history_indices"]
        cached = {frame:i for i,frame in enumerate([*history,*range(start-TAIL,start)])}
        frames = sorted(set(range(0,start,STRIDE)) | {start-1})
        selection = [cached[frame] for frame in frames]
        current = np.asarray(meta["camera_poses"][-4:])[:,[0,2,3]]
        scene = meta["layout"]
        for slot, arm in record["arms"].items():
            slot = int(slot)
            grid = torch.cat((arm["history"],record["shared"][:TAIL]))[selection]
            memory = CellMemory(poses,poses[frames],project(encoder,grid))
            route = RouteLookahead(current[0,:2],[*scene["candidate_routes"][slot],scene["candidate_positions"][slot]])
            waypoints = [route.target(pose) for pose in current]
            yield case(memory,scene,slot,current,waypoints,row)


@torch.inference_mode()
def predict(model, item):
    memory = item["memory"]
    entries, scores = model.encode(memory)
    goal = int(scores.argmax())
    chosen = int(nearest_cells(item["candidates"],memory.points[goal])[0])
    waypoints = model.waypoints(memory,entries,item["pose"],goal).argmax(-1)
    correct = item["waypoints"]
    error = (memory.points[waypoints]-memory.points[correct]).norm(dim=-1)
    return dict(seed=item["row"]["seed"],split=item["row"]["split"],distance=item["row"]["distance"],
                slot=item["slot"],prediction=chosen,goal_cell_correct=goal==item["goal"],
                waypoint_cells_correct=int((waypoints==correct).sum()),
                waypoint_within_two=int((error<=2).sum()),waypoint_n=len(error),
                waypoint_mean_error=float(error.mean()),goal_world=memory.points[goal].tolist())


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"],row["distance"]].append(row)
    metrics = {}
    for (split,distance), group in groups.items():
        choices = defaultdict(dict)
        for row in group:
            choices[row["seed"]][row["slot"]] = row["prediction"]
        paired = [pair for pair in choices.values() if set(pair)=={0,1}]
        n = len(group)
        correct = sum(r["slot"]==r["prediction"] for r in group)
        counts = {key:sum(r[key] for r in group) for key in
                  ("goal_cell_correct","waypoint_cells_correct","waypoint_within_two","waypoint_n")}
        metrics[f"{split}/{distance}"] = dict(n=n,assigned_right_room=correct,room_wilson95=wilson(correct,n),
            pairs=len(paired),flips=sum(pair[0]!=pair[1] for pair in paired),
            both_correct=sum(pair[0]==0 and pair[1]==1 for pair in paired),**counts)
    return metrics


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    checkpoint = torch.load(args.room_weights,map_location="cpu",weights_only=True)
    encoder = RoomReadout(checkpoint["specification"])
    encoder.load_state_dict(checkpoint["room_state"])
    encoder.eval().requires_grad_(False)
    training_cases = list(training(args.train_plan,args.train_cache,encoder))
    holdouts = list(heldout(args.holdout,args.holdout_cache,encoder))
    validation = [item for item in holdouts if item["row"]["split"]=="validation"]
    for key in ("seed","world"):
        if {item["row"][key] for item in training_cases}&{item["row"][key] for item in holdouts}:
            raise ValueError(f"Training and held-out {key} overlap")
    print("Loaded training",dict(Counter(x["row"]["distance"] for x in training_cases)),
          "held-out arms",dict(Counter(x["row"]["split"] for x in holdouts)),flush=True)
    width = encoder.reader.width-32
    model = CellPointers(width)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    best = None
    args.output.mkdir(exist_ok=True)
    with (args.output/"training.csv").open("x",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch","goal_loss","point_loss","validation_rooms","validation_goal_cells",
                         "validation_waypoints_within_two"))
        for epoch in range(1,args.epochs+1):
            model.train()
            rng.shuffle(training_cases)
            goal_loss = point_loss = 0.
            for item in training_cases:
                memory = item["memory"]
                entries, logits = model.encode(memory)
                goal = item["goal"]
                goal_objective = F.cross_entropy(logits[None],torch.tensor([goal]))
                count = len(item["pose"])
                start = (epoch-1)*48 % count
                subset = torch.arange(start,start+min(48,count)) % count
                point_logits = model.waypoints(memory,entries,item["pose"][subset],goal)
                point_objective = F.cross_entropy(point_logits,item["waypoints"][subset])
                optimizer.zero_grad(set_to_none=True)
                (goal_objective+point_objective).backward()
                optimizer.step()
                goal_loss += float(goal_objective.detach())
                point_loss += float(point_objective.detach())
            model.eval()
            checks = [predict(model,item) for item in validation]
            rank = (sum(r["slot"]==r["prediction"] for r in checks),
                    sum(r["waypoint_within_two"] for r in checks),sum(r["goal_cell_correct"] for r in checks))
            output = (epoch,goal_loss/len(training_cases),point_loss/len(training_cases),rank[0],rank[2],rank[1])
            writer.writerow(output)
            stream.flush()
            print("epoch",output,flush=True)
            if best is None or rank>best:
                best = rank
                temp = args.output/"pointer.partial"
                torch.save(dict(visual_width=width,state=model.state_dict(),epoch=epoch,
                                room_weights_sha256=hashlib.sha256(args.room_weights.read_bytes()).hexdigest()),temp)
                temp.replace(args.output/"pointer.pt")
    model.load_state_dict(torch.load(args.output/"pointer.pt",weights_only=True)["state"])
    rows = [predict(model,item) for item in holdouts]
    metrics = dict(train_trajectories=len(training_cases),train_supervised_states=sum(len(x["pose"]) for x in training_cases),
                   validation_arms=len(validation), groups=summarize(rows))
    for name,value in (("predictions.json",rows),("offline-results.json",metrics)):
        (args.output/name).write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps(metrics),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-plan","train-cache","holdout","holdout-cache","room-weights","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=24)
    parser.add_argument("--seed",type=int,default=20261128)
    train(parser.parse_args())

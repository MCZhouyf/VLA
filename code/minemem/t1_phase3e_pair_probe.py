"""Train matched v3 readout heads and score the frozen material-swap holdouts."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase1_metrics import wilson
from t1_spatial import GeometricSpatialMemory, QUERIES, spatial_read_plan
from t1_storage import retry_read
from t1_waypoint import DIRECTION_BINS


VARIANTS = ("current", "long", "spatial")
CONTROLS = ("shuffle_visual", "shuffle_pose")
DISTANCES = (10, 30, 60, 100)
STATES = 4
TAIL = 8
DISTANCE_BINS = 17


def window_plan(history,states=STATES):
    current = np.arange(TAIL,TAIL+states)
    offsets = np.rint(np.linspace(history,1,QUERIES)).astype(np.int64) if history else np.zeros(QUERIES,np.int64)
    index = np.repeat((current[:,None]-offsets)[...,None],2,axis=-1)
    geometry = np.zeros((*index.shape,9),dtype=np.float32)
    geometry[...,6] = np.log1p(offsets)[None,:,None]/8
    return index,geometry,np.zeros(index.shape,dtype=np.float32)


def history_plan(poses, current, side, anchors):
    plans = [spatial_read_plan(poses,pose,side,room_anchors=anchors) for pose in current]
    return tuple(np.stack(parts) for parts in zip(*plans,strict=True))


def model_case(record, side, device, interventions):
    poses = np.asarray(record["history_pose"],dtype=np.float32)
    current = np.asarray(record["poses"],dtype=np.float32)
    anchors = np.asarray(record["anchors"],dtype=np.int64)
    if current.shape != (STATES,3) or len(set(anchors))!=2:
        raise ValueError(f"Incomplete four-state example {record['seed']}")
    read = {"current":window_plan(0),"long":window_plan(TAIL),
            "spatial":history_plan(poses,current,side,anchors)}
    if interventions:
        perm = np.random.default_rng(record["seed"]).permutation(len(poses))
        idx,geo,bias = read["spatial"]
        read["shuffle_visual"] = perm[idx],geo,bias
        inverse = np.argsort(perm)
        read["shuffle_pose"] = history_plan(poses[perm],current,side,inverse[anchors])
    plans = {k:tuple(torch.from_numpy(part).to(device) for part in plan) for k,plan in read.items()}
    common = {k:record[k] for k in ("split","seed","world","distance","gold")}
    common.update(shared=record["shared"].to(device),plans=plans)
    return [dict(**common,slot=int(slot),history=item["history"].to(device),
                 labels=torch.tensor(item["labels"],dtype=torch.long,device=device))
            for slot,item in record["arms"].items()]


class RoomReadout(nn.Module):
    def __init__(self, specification):
        super().__init__()
        self.reader = GeometricSpatialMemory(specification["vision_dim"],specification["embedding_dim"],
                                             specification["embedding_norm"],specification["patch_side"],
                                             grid_side=specification["grid_side"])
        self.reader.to_llm.requires_grad_(False)
        width = self.reader.width
        self.candidate = nn.Sequential(nn.Linear(width*3-32,64),nn.LayerNorm(64),
                                       nn.GELU(),nn.Linear(64,2))
        self.waypoint = nn.Sequential(nn.Linear(QUERIES*width+width-32,128),nn.LayerNorm(128),
                                      nn.GELU(),nn.Linear(128,DIRECTION_BINS+DISTANCE_BINS))
        self.register_buffer("mean",specification["mean"])
        self.register_buffer("std",specification["std"])

    def embed(self, grid):
        return self.reader.encode_visual(((grid.float()-self.mean)/self.std).clamp(-8,8))

    def forward(self, case, variant):
        window = self.embed(case["shared"])
        current = window[TAIL:]
        storage = window if variant in ("current","long") else self.embed(case["history"])
        indices,geometry,bias = case["plans"][variant]
        readout = self.reader.read_projected(storage[indices],geometry,bias)
        candidate = self.candidate(torch.cat((current,readout[:,:2].flatten(1)),dim=-1))
        waypoint = self.waypoint(torch.cat((current,readout.flatten(1)),dim=-1))
        direction,distance = waypoint.split((DIRECTION_BINS,DISTANCE_BINS),dim=-1)
        return candidate,direction,distance


def dataset_stats(records):
    count = 0
    total = squares = None
    for record in records:
        history, = record["arms"].values()
        frame = torch.cat((history["history"],record["shared"])).float().flatten(0,1)
        if total is None:
            total = torch.zeros(frame.shape[-1],dtype=torch.float64)
            squares = torch.zeros_like(total)
        count += len(frame)
        total += frame.sum(0,dtype=torch.float64)
        squares += frame.square().sum(0,dtype=torch.float64)
    mean=total/count
    return mean.float(),(squares/count-mean.square()).clamp_min(1e-4).sqrt().float()


def train_rows(source):
    rows = json.loads(source.read_text())["episodes"]
    return [row for row in rows if row["role"] in ("train","dart") and row["group"] in DISTANCES
            and row["selected_labels"] >= STATES]


def load_cache(cache, seed):
    return retry_read(partial(torch.load,map_location="cpu",weights_only=True,mmap=True),cache/f"{seed}.pt")


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output.mkdir()
    initial = torch.load(args.initial,map_location="cpu",weights_only=True)
    records = []
    for row in train_rows(args.train_plan):
        item = load_cache(args.cache,row["scene_seed"])
        if item["seed"] != row["scene_seed"] or item["split"] != "train" or item["world"] != row["world_seed"]:
            raise ValueError("The training-only feature index has changed")
        records.append(item)
    side = round(records[0]["shared"].shape[1]**.5)
    mean,std = dataset_stats(records)
    specification = {key:initial[key] for key in ("vision_dim","embedding_dim","embedding_norm","grid_side")}
    specification.update(mean=mean,std=std,patch_side=side)
    first = RoomReadout(specification)
    expected = first.reader.state_dict()
    pretrained = {k.removeprefix("memory."):v for k,v in initial["model"].items() if k.startswith("memory.")}
    compatible = {k:v for k,v in pretrained.items() if k in expected and expected[k].shape==v.shape}
    first.reader.load_state_dict(compatible,strict=False)
    models = {variant:deepcopy(first).to(args.device) for variant in VARIANTS}
    train_set = [model_case(record,specification["grid_side"],args.device,False)[0] for record in records]
    if any(case["labels"].shape!=(STATES,2) or int(case["labels"][:,1].max()) >= DISTANCE_BINS for case in train_set):
        raise ValueError("Teacher waypoint label is out of range")
    configuration = dict(episodes=len(train_set),pool_side=side,grid_side=specification["grid_side"],
                         epochs=args.epochs,seed=args.seed,initial=str(args.initial),
                         preinitialised_tensors=sorted(compatible),supervised_states=STATES,
                         variants=VARIANTS,selected_training_seeds=sorted(case["seed"] for case in train_set))
    (args.output/"config.json").write_text(json.dumps(configuration,indent=2)+"\n")
    optim = torch.optim.AdamW([p for model in models.values() for p in model.parameters() if p.requires_grad],
                              lr=args.learning_rate,weight_decay=.003)
    with (args.output/"training.csv").open("x",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("epoch",*VARIANTS))
        for epoch in range(1,args.epochs+1):
            rng.shuffle(train_set)
            totals=np.zeros(len(VARIANTS))
            for case in train_set:
                target=torch.full((STATES,),case["slot"],device=args.device)
                losses=[]
                for i,(name,model) in enumerate(models.items()):
                    room,direction,distance=model(case,name)
                    loss=(F.cross_entropy(room,target)+F.cross_entropy(direction,case["labels"][:,0])
                          +F.cross_entropy(distance,case["labels"][:,1]))
                    losses.append(loss)
                    totals[i]+=float(loss.detach())
                optim.zero_grad(set_to_none=True)
                sum(losses).backward()
                optim.step()
            scores=(totals/len(train_set)).tolist()
            writer.writerow((epoch,*scores))
            stream.flush()
            print(f"epoch={epoch}/{args.epochs} loss={scores}",flush=True)
    torch.save(dict(specification=specification,models={k:v.state_dict() for k,v in models.items()}),
               args.output/"room-heads.pt")


def interval(values, paired, seed):
    total = sum(len(group) for group in values)
    correct = sum(sum(group) for group in values)
    if paired and correct in (0,total):
        limit=.025**(1/len(values))
        bounds=[limit,1.0] if correct else [0.0,1-limit]
    elif paired:
        groups = np.asarray([np.mean(group) for group in values])
        rng = np.random.default_rng(seed)
        resampled = groups[rng.integers(0,len(groups),size=(2000,len(groups)))].mean(axis=1)
        bounds = np.quantile(resampled,(.025,.975)).tolist()
    else:
        bounds = list(wilson(correct,total))
    return dict(correct=correct,n=total,rate=correct/total,ci95=bounds,clusters=len(values))


def summarise(predictions):
    buckets=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in predictions:
        buckets[row["split"],str(row["distance"]),row["state"],row["variant"]][row["seed"]][row["slot"]].append(row)
        buckets[row["split"],"all",row["state"],row["variant"]][row["world"]][row["seed"]].append(row)
    results={}
    for (split,distance,state,variant),batches in sorted(buckets.items()):
        paired=split=="test"
        groups=[]
        flips=[]
        for values in batches.values():
            flattened=[row for subset in values.values() for row in subset]
            groups.append(flattened)
            if paired:
                by_seed=defaultdict(list)
                for row in flattened:
                    by_seed[row["seed"]].append(row)
                for pair in by_seed.values():
                    if len(pair)!=2 or {r["slot"] for r in pair}!={0,1}:
                        raise ValueError("Broken paired prediction coverage")
                flips.append([int(pair[0]["room"] != pair[1]["room"]) for pair in by_seed.values()])
        clustered=paired or distance=="all"
        first=dict(room=interval([[int(r["room"]==r["slot"]) for r in group] for group in groups],
                                 clustered,20261126),
                   direction=interval([[int(r["direction"]==r["target_direction"]) for r in group] for group in groups],
                                      clustered,20261127))
        if paired:
            first["flip"]=interval(flips,distance=="all",20261128)
            primaries=[[r for r in group if r["primary"]] for group in groups]
            first["original_room"]=interval(
                [[int(r["room"]==r["slot"]) for r in group] for group in primaries],distance=="all",20261129)
        results[f"{split}/{distance}/{state}/{variant}"]=first
    return results


@torch.inference_mode()
def score(args):
    checkpoint = torch.load(args.weights,map_location="cpu",weights_only=True)
    specification=checkpoint["specification"]
    models={}
    for variant,weights in checkpoint["models"].items():
        head=RoomReadout(specification)
        head.load_state_dict(weights,strict=True)
        models[variant]=head.to(args.device).eval()
    records=json.loads(args.holdout.read_text())["archives"]
    predictions=[]
    for row in records:
        cached=load_cache(args.cache,row["seed"])
        if any(cached[k]!=row[k] for k in ("split","seed","world","distance")):
            raise ValueError(f"Scored scene is not in the frozen holdout: {row['seed']}")
        for case in model_case(cached,specification["grid_side"],args.device,True):
            for variant in (*VARIANTS,*CONTROLS):
                model=models["spatial" if variant in CONTROLS else variant]
                room,direction,_=model(case,variant)
                targets=case["labels"][:,0].tolist()
                for state,(r,d,t) in enumerate(zip(room.argmax(-1).tolist(),direction.argmax(-1).tolist(),targets,strict=True)):
                    predictions.append(dict(split=case["split"],seed=case["seed"],world=case["world"],
                                            distance=case["distance"],slot=case["slot"],
                                            primary=int(case["gold"])==case["slot"],state=state,variant=variant,
                                            room=r,direction=d,target_direction=t))
    results=summarise(predictions)
    same_targets=defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row["split"]=="test" and row["variant"]=="spatial":
            for distance in (str(row["distance"]),"all"):
                same_targets[f"{distance}/{row['state']}"][row["seed"]].append(row["target_direction"])
    label_agreement = {key:dict(same=sum(left==right for left,right in pairs.values()),n=len(pairs))
                       for key,pairs in same_targets.items()}
    groups={}
    for distance in (*DISTANCES,"all"):
        spatial=results[f"test/{distance}/0/spatial"]
        shuffled=results[f"test/{distance}/0/shuffle_visual"]
        room=spatial["room"]["rate"]
        flip=spatial["flip"]["rate"]
        improvement=room-shuffled["room"]["rate"]
        advantage=spatial["room"]["correct"]-shuffled["room"]["correct"]
        passed=(100*spatial["room"]["correct"]>=85*spatial["room"]["n"]
                and 100*spatial["flip"]["correct"]>=80*spatial["flip"]["n"]
                and 4*advantage>=spatial["room"]["n"])
        groups[str(distance)]=dict(room_accuracy=room,flip_rate=flip,
                                   improvement_over_shuffled_visual=improvement,passed=passed)
    args.output.mkdir()
    (args.output/"predictions.json").write_text(json.dumps(predictions,indent=2)+"\n")
    (args.output/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    summary=dict(primary="paired test at return state 0",passed=groups["all"]["passed"],distances=groups,
                 matched_waypoint_direction_labels=label_agreement,
                 weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest())
    (args.output/"offline_gate.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_subparsers(dest="stage",required=True)
    fit=stages.add_parser("train")
    infer=stages.add_parser("score")
    for stage in (fit,infer):
        for name in ("cache","output"):
            stage.add_argument(f"--{name}",type=Path,required=True)
        stage.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    for name in ("train-plan","initial"):
        fit.add_argument(f"--{name}",type=Path,required=True)
    fit.add_argument("--epochs",type=int,default=24)
    fit.add_argument("--seed",type=int,default=20261126)
    fit.add_argument("--learning-rate",type=float,default=.001)
    for name in ("holdout","weights"):
        infer.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    if args.device=="cuda" and torch.cuda.device_count()!=1:
        parser.error("Expose exactly one allocated GPU")
    (train if args.stage=="train" else score)(args)

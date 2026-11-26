"""Encode the same v3 observation windows for all paired offline controls."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image
from transformers import AutoProcessor

from t1_memory_cache import encode_image_batch, load_vision
from t1_storage import retry_read
from t1_waypoint import teacher_waypoints, waypoint_labels


RESOLUTION = (320, 180)
TAIL = 8
START_STATES = 4


class FrozenVision:
    def __init__(self, checkpoint, side, batch):
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one allocated GPU")
        self.encoder, _ = load_vision(checkpoint,"cuda")
        self.processor = AutoProcessor.from_pretrained(checkpoint,local_files_only=True).image_processor
        self.side = side
        self.batch = batch

    @torch.inference_mode()
    def frames(self, archive, names):
        features = []
        for start in range(0,len(names),self.batch):
            images = []
            for name in names[start:start+self.batch]:
                contents = retry_read(archive.read,name)
                with Image.open(io.BytesIO(contents)) as source:
                    images.append(source.convert("RGB").resize(RESOLUTION,Image.Resampling.BILINEAR))
            features.extend(grid for grid,_ in encode_image_batch(self.encoder,self.processor,images,self.side,"cuda"))
        return torch.stack(features)


def train_scene(row, local, old_cache, vision):
    original = Path(row["path"])
    cached = local/original.name
    old = old_cache/f"{row['scene_seed']}.pt"
    with retry_read(ZipFile,cached if cached.exists() else original) as archive:
        meta = json.loads(retry_read(archive.read,"episode.json"))
        scene = meta["scene"]
        if (scene["scenario_seed"],scene["world_seed"],meta["group"]) != (
                row["scene_seed"],row["world_seed"],row["group"]):
            raise ValueError(f"Training scene identity mismatch for {original}")
        start = meta["outbound_frames"]
        count = min(START_STATES,row["selected_labels"])
        if len(scene["room_frame_indices"]) != 2 or count < START_STATES:
            raise ValueError(f"Training episode has no full four-state room cue window: {original}")
        near = scene["room_frame_indices"]
        if any(index >= start-TAIL for index in near):
            raise ValueError(f"Training room observation is inside the short window: {original}")
        history = sorted(set(range(0,start-TAIL,8)) | set(near))
        window = [max(0,frame) for frame in range(start-TAIL,start+count)]
        frames = history+window
        previous = {}
        if vision.side==2 and old.exists():
            values = torch.load(old,map_location="cpu",weights_only=True,mmap=True)
            previous = {index:values["grid"][i] for i,index in enumerate(values["frame_indices"])}
        missing = sorted(set(frames)-previous.keys())
        if missing:
            features = vision.frames(archive,[f"{frame:05d}.jpg" for frame in missing])
            previous.update(zip(missing,features,strict=True))
        route = teacher_waypoints(meta)
        labels = [waypoint_labels(meta["pose"][start+i],route[i]) for i in range(count)]
        gold = str(scene["goal_slot"])
    return dict(split="train",seed=row["scene_seed"],world=row["world_seed"],distance=row["group"],gold=gold,
                shared=torch.stack([previous[i] for i in window]),poses=meta["pose"][start:start+count],
                history_pose=[meta["pose"][i] for i in history],anchors=[history.index(i) for i in near],
                arms={gold:dict(history=torch.stack([previous[i] for i in history]),labels=labels)})


def planar(pose):
    return [pose[0],pose[2],pose[3]]


def heldout_scene(row, vision):
    with retry_read(ZipFile,row["path"]) as archive:
        meta = json.loads(retry_read(archive.read,"episode.json"))
        if meta["row"]["seed"] != row["seed"] or meta["row"]["world"] != row["world"]:
            raise ValueError(f"Held-out scene identity mismatch for {row['seed']}")
        history = len(meta["history_indices"])
        window = vision.frames(archive,[f"window/{i:04d}.jpg" for i in range(TAIL+START_STATES)])
        arms = {}
        for slot in meta["slots"]:
            grid = vision.frames(archive,[f"goal{slot}/{i:04d}.jpg" for i in range(history)])
            arms[str(slot)] = dict(history=grid,labels=meta["labels"][str(slot)])
        poses = meta["rendered_poses"][str(meta["slots"][0])]
    return dict(split=row["split"],seed=row["seed"],world=row["world"],distance=row["distance"],gold=str(row["gold"]),
                shared=window,poses=[planar(pose) for pose in poses[-START_STATES:]],
                history_pose=[planar(pose) for pose in poses[:history]],anchors=meta["near_indices"],arms=arms)


def encode(args):
    args.output.mkdir(exist_ok=True)
    train = json.loads(args.train_plan.read_text())
    training = [row for row in train["episodes"] if row["role"] in ("train","dart")
                and row["group"] in (10,30,60,100) and row["selected_labels"] >= START_STATES]
    local = Path(train["archive_cache"])
    rows = [(row["scene_seed"],row,True) for row in training]
    if args.holdout:
        heldout = json.loads(args.holdout.read_text())["archives"]
        rows.extend((row["seed"],row,False) for row in heldout)
    if len({seed for seed,_,_ in rows}) != len(rows):
        raise ValueError("Training and holdout cache paths collide")
    assigned = rows[args.shard::2]
    vision = FrozenVision(args.checkpoint,args.pool_side,args.batch)
    for i,(seed,row,is_train) in enumerate(assigned,1):
        target = args.output/f"{seed}.pt"
        if target.exists():
            continue
        record = train_scene(row,local,args.old_cache,vision) if is_train else heldout_scene(row,vision)
        if record["shared"].shape[:2] != (TAIL+START_STATES,args.pool_side**2):
            raise ValueError(f"Unexpected shared window grid for {seed}")
        temporary = target.with_suffix(".partial")
        torch.save(record,temporary)
        temporary.replace(target)
        print(f"{i}/{len(assigned)} {record['split']} d={record['distance']} seed={seed} "
              f"anchors={record['anchors']}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","train-plan","old-cache","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--holdout",type=Path)
    parser.add_argument("--pool-side",type=int,choices=(2,4),default=2)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    parser.add_argument("--batch",type=int,default=12)
    args=parser.parse_args()
    if args.batch < 1:
        parser.error("Batch size must be positive")
    encode(args)

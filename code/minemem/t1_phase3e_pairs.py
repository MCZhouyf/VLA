"""Collect v3 holdouts and render material swaps at identical teacher poses."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np

from t1_dual_enclosure import candidate_occlusion, configure_rendering
from t1_phase3e import DISTANCES, FirstArrivalArena, frozen_gold_slots
from t1_pose_replay import return_by_pose
from t1_safety import POSE_FIELDS, StepTrace, life_values
from t1_shortcut_audit import gold_pixels
from t1_v4 import bearing, horizontal, wrapped_angle
from t1_waypoint import RouteLookahead, waypoint_labels


SCENE_BASE = 2135000000
WORLD_BASE = 21470000
SPLITS = ("validation", "test")
SCENES = 30
STRIDE = 8
TAIL = 8
START_STATES = 4


def plan_cases(formal, oracle_gate):
    gate = json.loads(oracle_gate.read_text())
    if not gate["passed"] or any(gate["distances"][str(d)]["n"] != 30
                                  or gate["distances"][str(d)]["successes"] < 24 for d in DISTANCES):
        raise ValueError("Stage A final oracle gate is required")
    index = formal / "index/dataset.jsonl"
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    summary = json.loads((formal / "index/summary.json").read_text())
    if digest != summary["index_sha256"]:
        raise ValueError("The frozen training index checksum has changed")
    fixtures = json.loads((formal / "frozen/fixtures.json").read_text())
    previous_worlds = {world for seeds in summary["world_seeds"].values() for world in seeds}
    previous_worlds.update(fixture["world_seed"] for fixture in fixtures["fixtures"].values())
    previous_scenes = set(fixtures["formal"])
    previous_scenes.update(json.loads(line)["scene_seed"] for line in index.read_text().splitlines())
    cases = []
    for split_index, split in enumerate(SPLITS):
        for group, distance in enumerate(DISTANCES):
            base = SCENE_BASE + split_index*1000000 + group*100
            for offset in range(64):
                seeds = [base + i*1000 + offset for i in range(SCENES)]
                try:
                    slots = frozen_gold_slots(seeds)
                except ValueError:
                    continue
                break
            else:
                raise ValueError(f"Could not prebalance layout and visit strata for {split}/{distance}")
            cases.extend(dict(split=split, ordinal=i, distance=distance, seed=seed, gold=slot,
                              world=WORLD_BASE + split_index*1000 + i)
                         for i,(seed,slot) in enumerate(zip(seeds,slots,strict=True)))
    if (any(row["seed"] in previous_scenes or row["world"] in previous_worlds for row in cases)
            or len({row["seed"] for row in cases}) != len(cases)):
        raise ValueError("Holdout scenes overlap earlier data")
    return dict(index_sha256=digest, oracle_gate=str(oracle_gate), cases=cases)


def pose_tuple(info):
    return tuple(float(info["player_pos"][key]) for key in POSE_FIELDS)


def encode_view(frame):
    success, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not success:
        raise RuntimeError("Could not encode the rendered observation")
    return jpeg.tobytes()


class PairedArena(FirstArrivalArena):
    def prepare_ground(self):
        # A long corridor can cross chunks outside the player's loaded radius.
        platform=f"{self.x-110} {self.z-110} {self.x+110} {self.z+110}"
        self.command(f"/forceload add {platform}")
        self.settle(32)
        try:
            super().prepare_ground()
        finally:
            self.command(f"/forceload remove {platform}")
        self.settle(6)

    def capture_frame(self):
        # EnclosureArena records an anchor when the teacher looks at each room.
        self.memory_frames.append(len(self.outbound))
        return len(self.memory_frames)-1

    def track_route(self, layout, error_trace):
        self.memory_frames = []
        trace = StepTrace(None)
        try:
            self.outward(layout, trace)
            near = [self.memory_frames[i] for i in layout["room_frame_indices"]]
        except RuntimeError:
            trace.path = error_trace
            trace.write()
            raise
        finally:
            self.memory_frames = None
            self.distractor = None
        poses = [tuple(row[1:6]) for row in trace.rows]
        if len(near) != 2 or len(poses) != len(self.outbound)+1 or trace.reason:
            raise RuntimeError("Canonical outbound trajectory is incomplete")
        self.restore_return_pose(layout)
        start = []
        for step in range(START_STATES):
            start.append(pose_tuple(self.info))
            if step+1 < START_STATES:
                current = horizontal(self.info)
                error = wrapped_angle(bearing(current, layout["corridor"])-self.info["player_pos"]["yaw"])
                self.teacher_step(max(-10, min(10, error)), abs(error) <= 8)
        return poses, near, start

    def material_slot(self, layout, slot):
        yaw = bearing(layout["spawn"], layout["corridor"])
        start = dict(zip(POSE_FIELDS,(layout["spawn"][0],self.y,layout["spawn"][1],yaw,0.0),strict=True))
        self.restore_return_pose({"return_pose":start})
        pillars = [tuple(map(math.floor,position)) for position in layout["candidate_positions"]]
        x,z = zip(*pillars,strict=True)
        self.command(f"/forceload add {min(x)} {min(z)} {max(x)} {max(z)}")
        self.settle(12)
        checks = []
        fills = []
        for candidate,(x,z) in enumerate(pillars):
            material = "gold_block" if candidate == slot else "lapis_block"
            fills.append((x,self.y,z,x,self.y+2,z,material))
            checks.extend((x,self.y+dy,z,material) for dy in (0,1,2))
        for attempt in range(3):
            for placement in fills:
                self.fill(*placement)
            self.settle(8)
            try:
                self.check_materials(checks)
            except RuntimeError:
                if attempt==2:
                    raise
                self.command("/clear @p minecraft:paper")
                self.settle(8)
            else:
                return

    def check_materials(self, checks):
        def wait_for_paper(present):
            for _ in range(32):
                self.tick()
                observed=any(slot["type"].split(":")[-1]=="paper" for slot in self.info["inventory"].values())
                if observed==present:
                    return
            raise RuntimeError(f"Material proof inventory did not synchronize (expected paper={present})")

        self.command("/clear @p minecraft:paper")
        wait_for_paper(False)
        for start in range(0,len(checks),3):
            group=checks[start:start+3]
            terms=" ".join(f"if block {x} {y} {z} minecraft:{material}" for x,y,z,material in group)
            self.command(f"/execute {terms} run give @p minecraft:paper 1")
            wait_for_paper(True)
            self.command("/clear @p minecraft:paper")
            wait_for_paper(False)

    def render_pose(self, pose):
        coords = " ".join(f"{value:.9f}" for value in pose)
        self.command(f"/tp @p {coords}")
        matched = 0
        for _ in range(24):
            self.tick()
            actual = pose_tuple(self.info)
            same_angles = all(abs(wrapped_angle(a-b)) <= 2e-4 for a,b in zip(actual[3:],pose[3:],strict=True))
            same_pose = math.dist(actual[:3],pose[:3]) <= 2e-4 and same_angles
            matched = matched+1 if same_pose else 0
            if matched == 2:
                health,alive,_ = life_values(self.info)
                if alive is False or (health is not None and health <= 0):
                    raise RuntimeError("The offline render did not preserve a live player")
                return actual,self.info["pov"]
        raise RuntimeError(f"Replay missed the canonical pose: requested={pose} actual={actual}")

    def teacher_return_trace(self, layout):
        trace = StepTrace(None)
        self.landmark,self.distractor = layout["landmark"],layout["distractor"]
        try:
            result = return_by_pose(self,layout,layout["teacher_targets"],trace,terminal_tolerance=.10)
        finally:
            self.distractor = None
        if not result[0] or trace.reason:
            raise RuntimeError(f"Canonical teacher return failed: {result[-1]}")
        return [tuple(row[1:6]) for row in trace.rows]

    def clear_navigation_corridor(self, layout):
        x,z = horizontal(self.info)
        dx,dz = self.direction
        cell=(math.floor(x+.55*dx),math.floor(z+.55*dz))
        sample=[(cell[0],self.y+level,cell[1],"air") for level in (0,1,2)]
        try:
            self.check_materials(sample)
        except RuntimeError:
            was_air=False
        else:
            was_air=True
        start,end=layout["corridor"],layout["endpoint"]
        x0,z0=(math.floor(min(a,b))-1 for a,b in zip(start,end,strict=True))
        x1,z1=(math.floor(max(a,b))+1 for a,b in zip(start,end,strict=True))
        self.command(f"/forceload add {x0} {z0} {x1} {z1}")
        self.settle(24)
        self.fill(x0,self.y-1,z0,x1,self.y-1,z1,"grass_block")
        self.fill(x0,self.y,z0,x1,self.y+5,z1,"air")
        self.settle(16)
        self.check_materials([*sample,(cell[0],self.y-1,cell[1],"grass_block")])
        return dict(contact_column=cell,air_check_before=was_air,air_check_after=True,
                    bounds=(x0,self.y-1,z0,x1,self.y+5,z1))


def return_labels(layout, slot, pose):
    route = RouteLookahead((pose[0][0],pose[0][2]),
                           [*layout["candidate_routes"][slot],layout["candidate_positions"][slot]])
    positions = [(x,z,yaw) for x,_,z,yaw,_ in pose]
    return [waypoint_labels(p,route.target(p)) for p in positions]


def capture(arena, row, output):
    layout = arena.setup(row["seed"],row["distance"],row["gold"])
    failure = output/f"{row['seed']}-outbound-error.csv.gz"
    try:
        poses,near,prefix = arena.track_route(layout,failure)
    except RuntimeError as error:
        if not str(error).startswith("Outbound teacher missed corridor waypoint"):
            raise
        clearing=arena.clear_navigation_corridor(layout)
        print(f"Revalidated original corridor for {row['seed']}: {clearing}",flush=True)
        layout=arena.setup(row["seed"],row["distance"],row["gold"])
        layout["corridor_repair"]=clearing
        poses,near,prefix=arena.track_route(layout,failure.with_name(f"{row['seed']}-outbound-retry-error.csv.gz"))
    teacher_return = arena.teacher_return_trace(layout)
    distant = layout["distant_frame_indices"]
    tail_start = len(poses)-TAIL-1
    if any(i >= tail_start for i in (*near,*distant)):
        raise RuntimeError("A pillar observation overlaps the final eight outbound frames")
    frames = sorted(set(range(0,tail_start,STRIDE)) | set(near) | set(distant))
    shared = poses[tail_start:tail_start+TAIL]+prefix
    for p in shared:
        geometry = dict(layout, return_pose=dict(zip(POSE_FIELDS,p,strict=True)))
        if not all(candidate_occlusion(geometry,arena.y)):
            raise RuntimeError("A candidate is exposed in the eight-frame window or return prefix")
    camera_poses = [poses[i] for i in frames]+shared
    slots = (row["gold"],1-row["gold"]) if row["split"] == "test" else (row["gold"],)
    recordings = {}
    pose_logs = {}
    view_counts = {}
    for slot in slots:
        arena.material_slot(layout,slot)
        observed,images = [],[]
        for index,pose in enumerate(camera_poses):
            actual,image = arena.render_pose(pose)
            observed.append(actual)
            if index in (frames.index(distant[slot]),frames.index(near[slot])):
                x,z = map(math.floor, layout["candidate_positions"][slot])
                area = gold_pixels(image,None,actual,(x,arena.y,z,x,arena.y+2,z),layout["occlusion_boxes"])
                view_counts.setdefault(slot,[]).append(area)
            images.append(encode_view(image))
        if len(view_counts[slot]) != 2 or min(view_counts[slot]) == 0:
            raise RuntimeError(f"Unrendered gold observation for slot {slot}: {view_counts[slot]}")
        recordings[slot],pose_logs[slot] = images,observed
    if len(slots) == 2 and pose_logs[slots[0]] != pose_logs[slots[1]]:
        raise RuntimeError("Material swap changed a recorded pose")
    hashes = {slot:[hashlib.sha256(frame).hexdigest() for frame in images[-len(shared):]]
              for slot,images in recordings.items()}
    shared_slot = slots[0] if len(slots)==1 else row["seed"]%2
    shared_images = recordings[shared_slot][-len(shared):]
    meta = dict(schema="phase3e_v3_swap",row=row,layout=layout,canonical_outbound_pose=poses,
                canonical_teacher_return_pose=teacher_return,
                camera_poses=camera_poses,rendered_poses=pose_logs,history_indices=frames,
                near_indices=[frames.index(i) for i in near],distant_indices=[frames.index(i) for i in distant],
                distant_sampling="teacher_dwell_start",
                slots=slots,shared_window_slot=shared_slot,rendered_window_sha256=hashes,
                shared_window_sha256=[hashlib.sha256(frame).hexdigest() for frame in shared_images],
                goal_view_pixels=view_counts, labels={slot:return_labels(layout,slot,prefix) for slot in slots})
    path = output / f"{row['seed']}.zip"
    temp = path.with_suffix(".partial")
    with ZipFile(temp,"w",compression=ZIP_STORED) as archive:
        archive.writestr("episode.json",json.dumps(meta,separators=(",",":")))
        for i,image in enumerate(shared_images):
            archive.writestr(f"window/{i:04d}.jpg",image)
        for slot,images in recordings.items():
            for i,image in enumerate(images[:len(frames)]):
                archive.writestr(f"goal{slot}/{i:04d}.jpg",image)
    temp.replace(path)
    print(f"{row['split']} d={row['distance']} seed={row['seed']} slots={slots} history={len(frames)} "
          f"gold_pixels={view_counts}",flush=True)


def collect(output, plan, worker, workers, limit, distances, previous):
    configure_rendering(output)
    cases = json.loads(plan.read_text())["cases"]
    completed = {int(path.stem) for root in previous
                 for path in (root/f"worker-{worker:03d}").glob("*.zip")}
    for split in SPLITS:
        for i in range(worker,limit,workers):
            rows = [row for row in cases if row["split"]==split and row["ordinal"]==i
                    and row["distance"] in distances and row["seed"] not in completed
                    and not (output/f"{row['seed']}.zip").exists()]
            if not rows:
                continue
            arena = PairedArena(rows[0]["world"],effects=True)
            try:
                for row in rows:
                    capture(arena,row,output)
            finally:
                arena.sim.close()


def audit(plan, roots, output):
    cases = {row["seed"]: row for row in json.loads(plan.read_text())["cases"]}
    archives = {}
    for root in roots:
        for path in root.glob("worker-*/*.zip"):
            seed = int(path.stem)
            if seed not in cases or seed in archives:
                raise ValueError(f"Unexpected or duplicated result {path}")
            archives[seed] = path
    if archives.keys() != cases.keys():
        raise ValueError(f"Missing holdout seeds {sorted(cases.keys()-archives.keys())}")
    rows = []
    for seed,row in sorted(cases.items()):
        path = archives[seed]
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("episode.json"))
            expected_slots = [row["gold"],1-row["gold"]] if row["split"]=="test" else [row["gold"]]
            if (manifest["row"] != row or manifest["slots"] != expected_slots
                    or not manifest["canonical_teacher_return_pose"]):
                raise ValueError(f"Holdout identity mismatch for {seed}")
            poses = manifest["rendered_poses"]
            if any(poses[str(slot)] != poses[str(expected_slots[0])] for slot in expected_slots):
                raise ValueError(f"Paired render poses differ for {seed}")
            hashes = [hashlib.sha256(archive.read(f"window/{i:04d}.jpg")).hexdigest()
                      for i in range(TAIL+START_STATES)]
            if hashes != manifest["shared_window_sha256"] or any(
                    min(manifest["goal_view_pixels"][str(slot)]) <= 0 for slot in expected_slots):
                raise ValueError(f"Bad observation or hidden window for {seed}")
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda:stream.read(1<<20),b""):
                digest.update(chunk)
        rows.append(dict(**row,path=str(path),sha256=digest.hexdigest(),paired=len(expected_slots)==2,
                         history_frames=len(manifest["history_indices"]),
                         goal_view_pixels=manifest["goal_view_pixels"],paired_poses_equal=len(expected_slots)==2))
    output.write_text(json.dumps(dict(archives=rows),indent=2)+"\n")
    print(f"Verified {len(rows)} primary trajectories and {sum(row['paired'] for row in rows)} twins",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--formal",type=Path)
    parser.add_argument("--oracle-gate",type=Path)
    parser.add_argument("--previous",nargs="*",type=Path,default=[])
    parser.add_argument("--audit",nargs="+",type=Path)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--workers",type=int)
    parser.add_argument("--limit",type=int,choices=range(1,SCENES+1),default=SCENES)
    parser.add_argument("--distances",type=int,nargs="+",choices=DISTANCES,default=DISTANCES)
    args = parser.parse_args()
    if args.formal is not None:
        if args.oracle_gate is None:
            parser.error("Building a plan requires the final oracle gate")
        args.output.write_text(json.dumps(plan_cases(args.formal,args.oracle_gate),indent=2)+"\n")
    elif args.audit is not None:
        if args.plan is None:
            parser.error("Auditing requires a frozen plan")
        audit(args.plan,args.audit,args.output)
    else:
        if args.plan is None or args.worker is None or not 0 <= args.worker < args.workers <= 32:
            parser.error("Supply a plan and valid worker shard")
        collect(args.output,args.plan,args.worker,args.workers,args.limit,args.distances,args.previous)

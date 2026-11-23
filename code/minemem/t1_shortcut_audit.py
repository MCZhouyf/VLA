"""Measure frozen T1 visibility and run position-free RGB controls."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from phase1_metrics import wilson
from phase1_t1_eval import fixture_index
from t1_collect import NAV_WORLD_SEED, ProtectedCaptureArena, navigation_cases
from t1_occlusion import SCREENED_PROTOCOL, OcclusionGeometry, set_candidate_render_distance
from t1_respawn import record_config, shortcut, verify_scene
from t1_safety import ProtectedArena, StepTrace, TrialInterrupted
from t1_v4 import DISTANCES, PROTOCOL, SUCCESS_RADIUS, bearing, horizontal, wrapped_angle


def gold_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (13, 104, 102), (37, 255, 255))
    mask[int(len(mask) * 0.80):] = 0
    return mask


def regions(mask, minimum=4):
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(int(area), float(x), float(y), int(height))
            for (_, _, _, height, area), (x, y) in zip(stats[1:count], centroids[1:count])
            if area >= minimum]


def gold_target(frame):
    return max(regions(gold_mask(frame)), default=(0, 0.0, 0.0, 0))


def blocked(origin, corners, box):
    lower, upper = np.asarray(box[:3]), np.asarray(box[3:])+1
    rays = corners - origin
    parallel = abs(rays) < 1e-9
    off_axis = np.any(parallel & ((origin<lower) | (origin>upper)),axis=1)
    denom = np.where(parallel,1,rays)
    a,b = (lower-origin)/denom, (upper-origin)/denom
    enters = np.where(parallel,-np.inf,np.minimum(a,b)).max(axis=1)
    exits = np.where(parallel,np.inf,np.maximum(a,b)).min(axis=1)
    return np.all(~off_axis & (enters<1) & (exits>=np.maximum(enters,0)))


def gold_pixels(frame, without_gold, pose, pillar, occluders=()):
    x,y,z = pillar[:3]
    corners = np.asarray(list(product((x,x+1),(y,y+3),(z,z+1))),dtype=float)
    origin = np.asarray((pose[0],pose[1]+1.62,pose[2]))
    if any(blocked(origin,corners,box) for box in occluders):
        return 0
    dx,dy,dz = (corners-origin).T
    yaw,pitch = map(math.radians,pose[3:])
    lateral = -dx*math.cos(yaw)-dz*math.sin(yaw)
    horizontal_forward = dz*math.cos(yaw)-dx*math.sin(yaw)
    depth = horizontal_forward*math.cos(pitch)-dy*math.sin(pitch)
    vertical = horizontal_forward*math.sin(pitch)+dy*math.cos(pitch)
    ahead = depth>0.01
    if not ahead.any():
        return 0
    h,w = frame.shape[:2]
    focal = h/(2*math.tan(math.radians(35)))
    px = w/2+focal*lateral[ahead]/depth[ahead]
    py = h/2-focal*vertical[ahead]/depth[ahead]
    # Allow small interpolation and pixel-center differences at the silhouette.
    x0,x1 = max(0,math.floor(px.min())-7), min(w,math.ceil(px.max())+8)
    y0,y1 = max(0,math.floor(py.min())-7), min(h,math.ceil(py.max())+8)
    if x0>=x1 or y0>=y1:
        return 0
    frame = frame[y0:y1,x0:x1]
    mask = cv2.inRange(cv2.cvtColor(frame,cv2.COLOR_RGB2HSV),(13,104,102),(37,255,255))
    if without_gold is not None:
        control = without_gold[y0:y1,x0:x1]
        mask[cv2.absdiff(frame,control).max(axis=2)<=24] = 0
    return sum(item[0] for item in regions(mask))


def wall_target(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    spread = frame.max(axis=2)-frame.min(axis=2)
    mask = ((spread <= 17) & (grey >= 42) & (grey <= 174)).astype(np.uint8)
    mask[:int(len(mask)*0.23)] = 0
    mask[int(len(mask)*0.78):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return max(regions(mask, 18), default=(0, 0.0, 0.0, 0))


def yaw_to_pixel(frame, point):
    width = frame.shape[1]
    return math.degrees(math.atan((point - width / 2) / (0.45 * width)))


def scan_policy(frame, move):
    """Choose controls from RGB and the action count only."""
    side = 1
    while True:
        gold = wall = (0, 0.0)
        for index in range(36):
            area, x, _, _ = gold_target(frame)
            if area > gold[0]:
                gold = area, wrapped_angle(10*index + yaw_to_pixel(frame, x))
            area, x, _, _ = wall_target(frame)
            if area > wall[0]:
                wall = area, wrapped_angle(10*index + yaw_to_pixel(frame, x))
            frame = move(yaw=10)
        area, direction = gold if gold[0] else wall
        while abs(direction) > 0.1:
            angle = max(-10, min(10, direction))
            frame = move(yaw=angle)
            direction -= angle
        if gold[0]:
            lost = 0
            for _ in range(512):
                pixels, x, _, _ = gold_target(frame)
                lost = 0 if pixels else lost + 1
                if lost >= 6:
                    break
                offset = yaw_to_pixel(frame, x) if pixels else 0.0
                frame = move(yaw=max(-10, min(10, offset)), forward=abs(offset) < 15)
        elif area:
            for _ in range(184):
                pixels, x, _, height = wall_target(frame)
                if gold_target(frame)[0] or not pixels or height >= frame.shape[0]*0.37:
                    break
                offset = yaw_to_pixel(frame, x)
                frame = move(yaw=max(-10, min(10, offset)), forward=abs(offset) < 15)
            for _ in range(43):
                frame = move(strafe=side)
                if gold_target(frame)[0]:
                    break
            for _ in range(54):
                if gold_target(frame)[0]:
                    break
                frame = move(forward=True)
            side = -side
        else:
            for _ in range(48):
                if gold_target(frame)[0]:
                    break
                frame = move(forward=True)


class TrialEnd(Exception):
    pass


def evaluate_control(arena, config, trace, scan, *, policy=None):
    arena.restore_return_pose(config)
    closest = math.dist(horizontal(arena.info), arena.landmark)
    reason = "budget"
    steps = 0

    def move(yaw=0.0, forward=False, strafe=0):
        nonlocal closest, steps, reason
        if steps >= config["return_limit"]:
            raise TrialEnd
        action = arena.sim.noop_action()
        action.update(camera=np.asarray([0.0, yaw], dtype=np.float32), forward=int(forward),
                      left=int(strafe < 0), right=int(strafe > 0))
        steps += 1
        if not arena.navigation_step(action):
            reason = trace.reason
            raise TrialEnd
        closest = min(closest, math.dist(horizontal(arena.info), arena.landmark))
        if closest <= SUCCESS_RADIUS:
            reason = "reached"
            raise TrialEnd
        return arena.info["pov"]

    try:
        with arena.track(trace):
            if policy is not None:
                policy(arena.info["pov"], move)
            elif scan:
                scan_policy(arena.info["pov"], move)
            else:
                while True:
                    move(forward=True)
    except TrialEnd:
        pass
    except TrialInterrupted as error:
        reason = str(error)
    return dict(success=reason=="reached", steps=steps, min_dist=closest, deaths=trace.deaths,
                pose_jumps=trace.pose_jumps, end_reason=reason)


def endpoint_views(arena, config):
    def frames():
        arena.restore_return_pose(config)
        start = float(arena.info["player_pos"]["yaw"])
        for index in range(36):
            arena.turn_to(wrapped_angle(start+10*index))
            pose = arena.info["player_pos"].copy()
            if math.dist((pose["x"], pose["z"]), horizontal({"player_pos": config["return_pose"]})) > 0.15:
                raise RuntimeError("Endpoint moved during visibility sweep")
            yield pose, arena.info["pov"].copy()

    rendered = list(frames())
    pillar = pillar_box(arena)
    arena.fill(*pillar, "air")
    try:
        paired = []
        for i, ((pose, frame), (control_pose, control)) in enumerate(zip(rendered, frames(), strict=True)):
            if abs(wrapped_angle(pose["yaw"]-control_pose["yaw"])) > 0.05:
                raise RuntimeError("Ablation render yaw differs from original")
            paired.append((i*10, pose["yaw"], gold_pixels(frame,control,coordinates(pose),pillar,walls(config))))
        return paired
    finally:
        arena.fill(*pillar, "gold_block")
        arena.settle()


def pillar_box(arena):
    return (arena.x, arena.y, arena.z, arena.x, arena.y+2, arena.z)


def walls(config):
    if config.get("occlusion_walls"):
        return (*config["occlusion_walls"],*config.get("pillar_screen",()))
    return (config["wall"],) if config.get("wall") else ()


def coordinates(pose):
    return tuple(float(pose[key]) for key in ("x","y","z","yaw","pitch"))


def confirmed_start(arena, config, occluders=()):
    arena.restore_return_pose(config)
    present = arena.info["pov"].copy()
    box = pillar_box(arena)
    arena.fill(*box, "air")
    try:
        arena.restore_return_pose(config)
        return gold_pixels(present,arena.info["pov"],coordinates(config["return_pose"]),box,occluders)
    finally:
        arena.fill(*box, "gold_block")
        arena.settle()


class ObservedTeacher(ProtectedArena):
    def __init__(self, world, effects):
        self.observe_return = False
        super().__init__(world, effects)

    def tick(self, action=None):
        super().tick(action)
        if self.observe_return:
            self.record_visibility()

    def record_visibility(self):
        step = len(self.visibility)
        pose = self.info["player_pos"].copy()
        area = gold_pixels(self.info["pov"],None,coordinates(pose),pillar_box(self),self.recorded_walls)
        self.visibility.append((step,math.dist(horizontal(self.info),self.landmark),area,pose))

    def teacher_visibility(self, fixture, output):
        self.recorded_walls = walls(fixture)
        targets = fixture.get("teacher_targets")
        if targets is None:
            if fixture["wall"]:
                raise RuntimeError("Missing verified return route from teacher fixture")
            targets = [self.landmark]
        self.restore_return_pose(fixture)
        self.visibility = []
        self.record_visibility()
        self.observe_return = True
        try:
            with self.track(StepTrace(None)):
                success, steps, _, _ = self.follow_return(targets,fixture["distance"])
            if not success or len(self.visibility)!=steps+1:
                raise RuntimeError("Could not replay selected teacher route")
        finally:
            self.observe_return = False
        first = None
        with output.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step","distance","candidate_gold_pixels","confirmed_gold_pixels"))
            for step,dist,pixels,pose in self.visibility:
                confirmed = ""
                if first is None and pixels:
                    confirmed = confirmed_start(self,{"return_pose":pose},self.recorded_walls)
                    if confirmed:
                        first = step,dist,confirmed
                writer.writerow((step,dist,pixels,confirmed))
        return first or ("","",0)


class OcclusionCandidate(OcclusionGeometry, ObservedTeacher):
    pass


def audit(args):
    frozen = json.loads(args.seeds_file.read_text())
    seeds = (frozen["formal"] if args.cohort=="formal" else
             sorted(map(int,frozen["world_seed_by_scenario"])))[args.worker::args.workers]
    if args.seed:
        seeds = [seed for seed in seeds if seed in args.seed]
    fixtures = fixture_index(args.fixtures)
    cases = set(map(tuple,args.case or ()))
    if args.candidate:
        set_candidate_render_distance(args.output)
    worlds = defaultdict(list)
    for seed in seeds:
        if not cases or any((d,seed) in cases for d in args.distances):
            worlds[frozen["world_seed_by_scenario"][str(seed)]].append(seed)
    fields = ("d","seed","world_seed","return_limit","any_visible","first_heading_pixels",
              "max_gold_pixels","visible_headings","teacher_first_step","teacher_first_dist",
              "teacher_first_pixels","success","steps","min_dist","deaths","pose_jumps","end_reason",
              "teacher_success","teacher_steps","turn180_success","turn180_deaths","turn180_pose_jumps",
              "outbound_error")
    if args.teacher_only:
        fields = ("d","seed","teacher_first_step","teacher_first_dist","teacher_first_pixels")
    if args.visibility_only:
        fields = fields[:11]
    with (args.output/"visibility.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for world, cohort in worlds.items():
            effects = fixtures[args.distances[0],cohort[0]]["optional_effects"]
            arena = (OcclusionCandidate if args.candidate else ObservedTeacher)(world, effects)
            if args.candidate:
                arena.screened_pillar = args.screened_pillar
            try:
                for seed in cohort:
                    for d in args.distances:
                        if cases and (d,seed) not in cases:
                            continue
                        fixture = fixtures[d,seed]
                        config = arena.prepare_episode(seed,d)
                        key = f"d{d}-seed{seed}"
                        extra = {}
                        reuse_candidate = args.candidate and args.visibility_only and fixture["protocol"]==SCREENED_PROTOCOL
                        if reuse_candidate:
                            for key_name in ("wall","occlusion_walls","pillar_screen","waypoints"):
                                if json.loads(json.dumps(config.get(key_name))) != fixture.get(key_name):
                                    raise ValueError(f"Candidate geometry changed before vision replay: {key_name}")
                            config = fixture.copy()
                        elif args.candidate:
                            outbound_error = ""
                            try:
                                with arena.track(StepTrace(args.output/f"{key}-outbound.csv.gz")):
                                    arena.outbound_segment(config)
                            except RuntimeError as error:
                                if not str(error).startswith("Teacher could not reach outbound waypoint"):
                                    raise
                                outbound_error = str(error)
                                x,z = config["planned_endpoint"]
                                heading = bearing((x,z),arena.landmark)
                                offset = random.Random(f"{PROTOCOL}:{seed}:{d}").uniform(60,300)
                                config.update(
                                    outbound_error=outbound_error,endpoint_source="planned_after_outbound_failure",
                                    return_pose=dict(x=x,y=arena.y,z=z,yaw=wrapped_angle(heading+offset),pitch=0.0),
                                )
                            arena.restore_return_pose(config)
                            reached, steps, _, _ = arena.teacher_return(config)
                            extra.update(teacher_success=reached and not outbound_error,teacher_steps=steps,
                                         outbound_error=outbound_error)
                            direct = shortcut(arena,config,StepTrace(args.output/f"{key}-180.csv.gz"))
                            extra.update(turn180_success=direct[0],turn180_deaths=direct[3],
                                         turn180_pose_jumps=direct[5])
                        else:
                            verify_scene(config, fixture)
                            config.update(return_pose=fixture["return_pose"], return_limit=fixture["return_limit"])
                        record_config(args.output, key, config)
                        if args.teacher_only:
                            step, dist, pixels = arena.teacher_visibility(fixture,args.output/f"{key}-teacher-confirmed.csv")
                            writer.writerow(dict(d=d,seed=seed,teacher_first_step=step,
                                                 teacher_first_dist=dist,teacher_first_pixels=pixels))
                            stream.flush()
                            continue
                        views = endpoint_views(arena, config)
                        with (args.output/f"{key}-sweep.csv").open("x", newline="") as output:
                            frame_writer=csv.writer(output)
                            frame_writer.writerow(("relative_yaw","actual_yaw","gold_pixels"))
                            frame_writer.writerows(views)
                        first_step, first_dist, first_pixels = ("","",0)
                        if not args.skip_teacher_visibility and extra.get("teacher_success",True):
                            first_step, first_dist, first_pixels = arena.teacher_visibility(
                                config if args.candidate else fixture, args.output/f"{key}-teacher.csv")
                        result = {}
                        if not args.visibility_only:
                            arena.command("/gamemode adventure @p")
                            arena.settle()
                            result = evaluate_control(arena, config,
                                                      StepTrace(args.output/f"{key}-rgb.csv.gz"), scan=True)
                            arena.command("/gamemode survival @p")
                            arena.settle()
                        pixels = [row[2] for row in views]
                        writer.writerow(dict(d=d, seed=seed, world_seed=world,
                                            return_limit=config["return_limit"], any_visible=max(pixels)>0,
                                            first_heading_pixels=pixels[0], max_gold_pixels=max(pixels),
                                            visible_headings=sum(p>0 for p in pixels),
                                            teacher_first_step=first_step, teacher_first_dist=first_dist,
                                            teacher_first_pixels=first_pixels, **result, **extra))
                        stream.flush()
                        print(d, seed, max(pixels), first_dist, result, flush=True)
            finally:
                arena.sim.close()


def navigation(args):
    fields = ("gate","seed","distance","heading_offset","cohort","gold_pixels_start",
              "success","steps","min_dist","deaths","pose_jumps","end_reason")
    arena = ProtectedCaptureArena(NAV_WORLD_SEED)
    try:
        with (args.output/"forward.csv").open("x",newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for offset in (False,True):
                for seed, distance, heading, cohort in navigation_cases(offset):
                    config = arena.prepare_visible(seed, distance, heading)
                    config["return_limit"] = 200
                    record_config(args.output, f"seed{seed}", config)
                    start_pixels = confirmed_start(arena,config)
                    result = evaluate_control(arena, config,
                                              StepTrace(args.output/f"seed{seed}-forward.csv.gz"), scan=False)
                    writer.writerow(dict(gate="offset" if offset else "original", seed=seed,
                                         distance=distance, heading_offset=heading, cohort=cohort,
                                         gold_pixels_start=start_pixels, **result))
                    stream.flush()
    finally:
        arena.sim.close()


def summarize(args):
    frozen = json.loads(args.seeds_file.read_text())
    expected = set(frozen["formal"] if args.cohort=="formal" else map(int,frozen["world_seed_by_scenario"]))
    data = defaultdict(dict)
    for path in args.inputs:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                d, seed = int(row["d"]), int(row["seed"])
                if seed in data[d]:
                    raise ValueError(f"Duplicate audit trial {d}/{seed}")
                data[d][seed] = row
    if set(data) != set(args.distances) or any(set(group) != expected for group in data.values()):
        raise ValueError("Visibility cohort is incomplete")
    if args.teacher_verified:
        merged = set()
        for path in args.teacher_verified:
            with path.open(newline="") as source:
                for row in csv.DictReader(source):
                    d, seed = int(row["d"]), int(row["seed"])
                    if (d,seed) in merged:
                        raise ValueError(f"Duplicate paired teacher result {d}/{seed}")
                    vision = ("any_visible","first_heading_pixels","max_gold_pixels","visible_headings")
                    data[d][seed].update({k:v for k,v in row.items() if k.startswith("teacher_first_") or k in vision})
                    merged.add((d,seed))
        if merged != {(d,seed) for d in args.distances for seed in expected}:
            raise ValueError("Paired teacher visibility cohort is incomplete")
    results = {}
    for d, group in sorted(data.items()):
        rows = list(group.values())
        n = len(rows)
        observed = [float(row["teacher_first_dist"]) for row in rows if row["teacher_first_dist"]]
        pix = [int(row["max_gold_pixels"]) for row in rows]
        hits = sum(row["success"]=="True" for row in rows)
        results[d] = dict(n=n, any_visible=sum(p>0 for p in pix),
                          visible_at_spawn_heading=sum(int(r["first_heading_pixels"])>0 for r in rows),
                          endpoint_pixel_quantiles=np.quantile(pix,[0,.25,.5,.75,1]).tolist(),
                          teacher_visible=len(observed),
                          teacher_first_dist_quantiles=np.quantile(observed,[0,.25,.5,.75,1]).tolist() if observed else [],
                          scan_success=hits, scan_rate=hits/n, scan_ci95=wilson(hits,n),
                          deaths=sum(int(r["deaths"]) for r in rows),
                          pose_jumps=sum(int(r["pose_jumps"]) for r in rows))
        if any(r.get("teacher_success") for r in rows):
            solved = [int(r["teacher_steps"]) for r in rows if r["teacher_success"]=="True"]
            results[d].update(teacher_success=len(solved),
                              mean_teacher_steps=float(np.mean(solved)) if solved else None,
                              outbound_failures=sum(bool(r.get("outbound_error")) for r in rows),
                              turn180_success=sum(r["turn180_success"]=="True" for r in rows),
                              turn180_deaths=sum(int(r["turn180_deaths"]) for r in rows),
                              turn180_pose_jumps=sum(int(r["turn180_pose_jumps"]) for r in rows))
    args.output.write_text(json.dumps(results,indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("audit")
    run.add_argument("output", type=Path)
    run.add_argument("--worker", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--seeds-file", type=Path, required=True)
    run.add_argument("--fixtures", type=Path, nargs="+")
    run.add_argument("--structure-scores", type=Path)
    run.add_argument("--structure-ablation-only", action="store_true")
    run.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=DISTANCES)
    run.add_argument("--candidate", action="store_true")
    run.add_argument("--screened-pillar", action="store_true")
    run.add_argument("--cohort", choices=("formal","teacher"), default="formal")
    run.add_argument("--seed", type=int, nargs="+")
    run.add_argument("--case", type=int, nargs=2, action="append", metavar=("DISTANCE","SEED"))
    run.add_argument("--teacher-only", action="store_true")
    run.add_argument("--visibility-only", action="store_true")
    run.add_argument("--skip-teacher-visibility", action="store_true")
    nav = modes.add_parser("navigation")
    nav.add_argument("output", type=Path)
    nav.add_argument("--worker", type=int, default=0)
    nav.add_argument("--workers", type=int, choices=(1,), default=1)
    report = modes.add_parser("summary")
    report.add_argument("output", type=Path)
    report.add_argument("--seeds-file", type=Path, required=True)
    report.add_argument("--inputs", type=Path, nargs="+", required=True)
    report.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=DISTANCES)
    report.add_argument("--cohort", choices=("formal","teacher"), default="formal")
    report.add_argument("--teacher-verified", type=Path, nargs="+")
    args = parser.parse_args()
    if args.mode == "audit" and args.structure_scores:
        from t1_structure_audit import audit_structures

        audit_structures(args)
        return
    if args.mode == "audit" and not args.fixtures:
        parser.error("The visibility audit needs --fixtures or --structure-scores")
    if args.mode == "audit" and args.structure_ablation_only:
        parser.error("--structure-ablation-only requires --structure-scores")
    if args.mode == "audit" and args.screened_pillar and not args.candidate:
        parser.error("A screened pillar requires candidate geometry")
    {"audit":audit, "navigation":navigation, "summary":summarize}[args.mode](args)


if __name__ == "__main__":
    main()

"""Measure visibility of the screened T1 structures and score an RGB-only shortcut."""

import argparse
import csv
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from phase1_metrics import wilson
from t1_occlusion import SCREENED_PROTOCOL, OcclusionGeometry, set_candidate_render_distance
from t1_respawn import record_config
from t1_safety import ProtectedArena, StepTrace, TrialInterrupted
from t1_shortcut_audit import evaluate_control, gold_target, regions, yaw_to_pixel
from t1_v4 import horizontal, wrapped_angle


STRUCTURES = ("endpoint_walls", "gold_enclosure")
BOX_EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if not i & bit]
FIELDS = (
    "d", "seed", "world_seed", "return_limit",
    "endpoint_walls_visible_headings", "gold_enclosure_visible_headings",
    "endpoint_walls_max_pixels", "gold_enclosure_max_pixels",
    "endpoint_walls_first_step", "endpoint_walls_first_dist", "endpoint_walls_first_pixels",
    "gold_enclosure_first_step", "gold_enclosure_first_dist", "gold_enclosure_first_pixels",
    "endpoint_source", "teacher_replay", "success", "steps", "min_dist", "deaths", "pose_jumps", "end_reason",
)


def stone_mask(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Cobblestone is textured grey. This mask is a material cue used by the
    # policy, so the paired evidence below separately checks attribution.
    return ((frame.max(axis=2) - frame.min(axis=2) <= 24)
            & (grey >= 37) & (grey <= 197)).astype(np.uint8)


def structure_target(frame):
    mask = stone_mask(frame)
    height = len(mask)
    mask[:int(height * .18)] = 0
    mask[int(height * .79):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return max(regions(mask, 45), default=(0, 0.0, 0.0, 0))


def look(frame, move, angle):
    angle = wrapped_angle(angle)
    while abs(angle) > 0.1:
        increment = max(-10, min(10, angle))
        frame = move(yaw=increment)
        angle -= increment
    return frame


def follow_gold(frame, move):
    lost = 0
    while lost < 5:
        pixels, x, _, _ = gold_target(frame)
        lost = 0 if pixels else lost + 1
        angle = yaw_to_pixel(frame, x) if pixels else 0
        frame = move(yaw=max(-10, min(10, angle)), forward=abs(angle) < 15)
    return frame


def entrance(frame):
    height, width = frame.shape[:2]
    facade = stone_mask(frame)[int(.25 * height):int(.63 * height)]
    columns = np.mean(facade, axis=0) >= .30
    edges = np.flatnonzero(np.diff(np.r_[True, columns, True]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if start < .07 * width or stop > .93 * width or not .09 * width <= stop - start <= .52 * width:
            continue
        wing = max(1, int(.05 * width))
        if columns[start-wing:start].mean() >= .8 and columns[stop:stop+wing].mean() >= .8:
            return (start + stop) / 2
    return None


def walk_facade(frame, move, side):
    """Strafe along the facing wall, then step around a visually cleared edge."""
    height, width = frame.shape[:2]
    for _ in range(95):
        if gold_target(frame)[0]:
            return frame
        door = entrance(frame)
        if door is not None:
            angle = yaw_to_pixel(frame, door)
            if abs(angle) < 6:
                for _ in range(31):
                    if gold_target(frame)[0]:
                        return frame
                    frame = move(forward=True)
                return frame
            frame = move(strafe=1 if angle > 0 else -1)
            continue
        area, center, _, span = structure_target(frame)
        if area < width * height * .026 or span < height * .20:
            break
        # The centroid reaches the far side of the view at a corner or opening.
        if (center - (width / 2)) * side < -width * .31:
            break
        frame = move(strafe=side)
    for _ in range(16):
        if gold_target(frame)[0]:
            return frame
        frame = move(strafe=side)
    for _ in range(37):
        if gold_target(frame)[0]:
            return frame
        area, center, _, span = structure_target(frame)
        if area > width * height * .18 and span > height * .47:
            break
        frame = move(forward=True)
    return look(frame, move, -side * 90)


def find_structure_policy(frame, move):
    """The only observations are RGB images and the actions issued so far."""
    side = 1
    familiar = None
    while True:
        gold = (0, 0.0)
        proposals = []
        for index in range(36):
            score, x, _, _ = gold_target(frame)
            if score > gold[0]:
                gold = score, wrapped_angle(10 * index + yaw_to_pixel(frame, x))
            score, x, _, _ = structure_target(frame)
            if score:
                proposals.append((score, wrapped_angle(10 * index + yaw_to_pixel(frame, x))))
            frame = move(yaw=10)
        if gold[0]:
            frame = follow_gold(look(frame, move, gold[1]), move)
            continue
        if not proposals:
            for _ in range(28):
                frame = move(forward=True)
                if gold_target(frame)[0]:
                    frame = follow_gold(frame, move)
                    break
            familiar = None
            continue
        novel = [item for item in proposals
                 if familiar is None or abs(wrapped_angle(item[1] - familiar)) > 35]
        _, heading = max(novel or proposals)
        frame = look(frame, move, heading)
        familiar = 0
        for _ in range(185):
            if gold_target(frame)[0]:
                frame = follow_gold(frame, move)
                break
            area, x, _, span = structure_target(frame)
            height, width = frame.shape[:2]
            if not area or span >= .47 * height or area >= .20 * height * width:
                break
            angle = yaw_to_pixel(frame, x)
            frame = move(yaw=max(-10, min(10, angle)), forward=abs(angle) < 18)
        for _ in range(3):
            frame = walk_facade(frame, move, side)
            if gold_target(frame)[0]:
                frame = follow_gold(frame, move)
                break
        side = -side


def structure_boxes(config):
    return dict(endpoint_walls=config["occlusion_walls"], gold_enclosure=config["pillar_screen"])


def projection(shape, pose, box):
    height, width = shape[:2]
    x, y, z = (pose[key] for key in ("x", "y", "z"))
    corners = np.asarray(list(product(*[(box[i], box[i + 3] + 1) for i in range(3)])))
    dx, dy, dz = (corners - (x, y + 1.62, z)).T
    yaw, pitch = (math.radians(pose[key]) for key in ("yaw", "pitch"))
    lateral = -dx * math.cos(yaw) - dz * math.sin(yaw)
    h_forward = dz * math.cos(yaw) - dx * math.sin(yaw)
    depth = h_forward * math.cos(pitch) - dy * math.sin(pitch)
    vertical = h_forward * math.sin(pitch) + dy * math.cos(pitch)
    near = .02
    ahead = depth >= near
    if not ahead.any():
        return None
    points = np.column_stack((lateral, vertical, depth))
    clipped = list(points[ahead])
    if not ahead.all():
        for i, j in BOX_EDGES:
            if ahead[i] != ahead[j]:
                clipped.append(points[i] + (points[j] - points[i]) * ((near - depth[i]) / (depth[j] - depth[i])))
    lateral, vertical, depth = np.asarray(clipped).T
    focal = height / (2 * math.tan(math.radians(35)))
    px = width / 2 + focal * lateral / depth
    py = height / 2 - focal * vertical / depth
    x0, x1 = max(0, math.floor(px.min()) - 7), min(width, math.ceil(px.max()) + 8)
    y0, y1 = max(0, math.floor(py.min()) - 7), min(height, math.ceil(py.max()) + 8)
    return (y0, y1, x0, x1) if x0 < x1 and y0 < y1 else None


def structure_pixels(frame, pose, boxes, control=None, material=None):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for box in boxes:
        rect = projection(frame.shape, pose, box)
        if rect:
            y0, y1, x0, x1 = rect
            mask[y0:y1, x0:x1] = 1
    if not mask.any():
        return 0
    mask &= stone_mask(frame) if material is None else material
    if control is not None:
        mask[cv2.absdiff(frame, control).max(axis=2) <= 24] = 0
        return sum(area for area, _, _, _ in regions(mask, minimum=12))
    return int(np.count_nonzero(mask))


@contextmanager
def without_structure(arena, boxes):
    for box in boxes:
        arena.fill(*box, "air")
    arena.settle()
    try:
        yield
    finally:
        for box in boxes:
            arena.fill(*box, "cobblestone")
        arena.settle()


def check_match(pose, other):
    if (math.dist([pose[k] for k in ("x", "y", "z")],
                  [other[k] for k in ("x", "y", "z")]) > .02
            or any(abs(wrapped_angle(pose[k] - other[k])) > .05 for k in ("yaw", "pitch"))):
        raise RuntimeError("Structure ablation views have different camera poses")


def endpoint_pairs(arena, config, groups, output):
    def capture():
        arena.restore_return_pose(config)
        start = arena.info["player_pos"].copy()
        check_match(config["return_pose"], start)
        yaw = start["yaw"]
        for index in range(36):
            heading = wrapped_angle(yaw + 10 * index)
            arena.turn_to(heading)
            pose = arena.info["player_pos"].copy()
            check_match(dict(start, yaw=heading), pose)
            yield pose, arena.info["pov"].copy()

    present = list(capture())
    measured = {}
    for name, boxes in groups.items():
        with without_structure(arena, boxes):
            measured[name] = []
            for (pose, frame), (other, control) in zip(present, capture(), strict=True):
                check_match(pose, other)
                measured[name].append(structure_pixels(frame, pose, boxes, control))
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_yaw", "actual_yaw", *groups))
        for index, (pose, _) in enumerate(present):
            writer.writerow((index * 10, pose["yaw"], *(measured[k][index] for k in groups)))
    return {key: value for name, counts in measured.items()
            for key, value in ((f"{name}_visible_headings", sum(x > 0 for x in counts)),
                               (f"{name}_max_pixels", max(counts)))}


class ScreenedStructureArena(OcclusionGeometry, ProtectedArena):
    screened_pillar = True
    teacher_candidates = None

    def restore_return_pose(self, config):
        super().restore_return_pose(config)
        # Closing out the teleport outside scored control prevents a short
        # server-side fall or heading snap from becoming an ablation difference.
        self.settle(16)

    def record_teacher_view(self):
        pose = self.info["player_pos"].copy()
        frame = self.info["pov"]
        material = stone_mask(frame)
        candidates = [structure_pixels(frame, pose, boxes, material=material) for boxes in self.groups.values()]
        distance = math.dist(horizontal(self.info), self.landmark)
        self.teacher_candidates.append((distance, pose, candidates))

    def navigation_step(self, action):
        valid = super().navigation_step(action)
        if valid and self.teacher_candidates is not None:
            self.record_teacher_view()
        return valid

    def verify_teacher_view(self, pose, boxes):
        config = {"return_pose": pose}
        self.restore_return_pose(config)
        frame = self.info["pov"].copy()
        measured_pose = self.info["player_pos"].copy()
        check_match(pose, measured_pose)
        with without_structure(self, boxes):
            self.restore_return_pose(config)
            check_match(measured_pose, self.info["player_pos"])
            return structure_pixels(frame, measured_pose, boxes, self.info["pov"])

    def teacher_structure_visibility(self, config, groups, output):
        if not config.get("teacher_targets"):
            return dict(teacher_replay="no_frozen_route")
        self.restore_return_pose(config)
        self.groups = groups
        self.teacher_candidates = []
        self.record_teacher_view()
        try:
            with self.track(StepTrace(None)):
                reached, _, _, _ = self.follow_return(config["teacher_targets"], config["distance"])
            replay = "reached" if reached else "route_failed"
        except TrialInterrupted as error:
            replay = str(error)
        finally:
            views = self.teacher_candidates
            self.teacher_candidates = None
        first = {}
        with output.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "distance", "x", "y", "z", "yaw", "pitch",
                             *[f"{key}_candidate" for key in groups],
                             *[f"{key}_paired" for key in groups]))
            for step, (distance, pose, candidates) in enumerate(views):
                row = []
                for (name, boxes), candidate in zip(groups.items(), candidates, strict=True):
                    pixels = ""
                    if candidate and name not in first:
                        pixels = self.verify_teacher_view(pose, boxes)
                        if pixels:
                            first[name] = (step, distance, pixels)
                    row.append(pixels)
                writer.writerow((step, distance, *(pose[k] for k in ("x", "y", "z", "yaw", "pitch")),
                                 *candidates, *row))
        return dict(teacher_replay=replay, **{
            f"{name}_first_{key}": value for name, values in first.items()
            for key, value in zip(("step", "dist", "pixels"), values, strict=True)
        })


def frozen_fixture_index(scores, wanted):
    selected = {}
    with scores.open(newline="") as source:
        for row in csv.DictReader(source):
            d, seed = int(row["d"]), int(row["seed"])
            case = d, seed
            if case not in wanted:
                continue
            if case in selected:
                raise ValueError(f"Duplicate acceptance score {case}")
            path = (scores.parent / row["source"]).parent / f"d{d}-seed{seed}.json"
            config = json.loads(path.read_text())
            if (config["distance"], config["scenario_seed"]) != case:
                raise ValueError(f"Acceptance source does not match the scored case {case}")
            if config["protocol"] != SCREENED_PROTOCOL or not config["screened_pillar"]:
                raise ValueError(f"Unexpected acceptance protocol at {path}")
            selected[case] = config
    if set(selected) != wanted:
        raise ValueError(f"Missing screened acceptance fixtures: {sorted(wanted - selected.keys())}")
    return selected


def audit_structures(args):
    frozen = json.loads(args.seeds_file.read_text())
    seeds = frozen["formal"] if args.cohort == "formal" else sorted(map(int, frozen["world_seed_by_scenario"]))
    cases = {(d, seed) for seed in seeds for d in args.distances if d >= 30}
    if args.seed:
        cases = {case for case in cases if case[1] in args.seed}
    if args.case:
        chosen = {tuple(case) for case in args.case}
        if not chosen <= cases:
            raise ValueError("Requested structure audit case is outside the selected cohort")
        cases &= chosen
    configs = frozen_fixture_index(args.structure_scores, cases)
    worlds = defaultdict(list)
    for case, config in sorted(configs.items()):
        world = frozen["world_seed_by_scenario"][str(case[1])]
        if config["world_seed"] != world:
            raise ValueError(f"Frozen world differs from the screened fixture for {case}")
        worlds[world].append(config)
    set_candidate_render_distance(args.output)
    with (args.output / "structure.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        for world in sorted(worlds)[args.worker::args.workers]:
            cohort = worlds[world]
            arena = ScreenedStructureArena(world, cohort[0]["optional_effects"])
            try:
                for config in cohort:
                    d, seed = config["distance"], config["scenario_seed"]
                    name = f"d{d}-seed{seed}"
                    generated = arena.prepare_episode(seed, d)
                    for key in ("protocol", "arena_origin", "landmark", "occlusion_walls",
                                "pillar_screen", "planned_endpoint", "waypoints", "render_distance_chunks",
                                "optional_effects"):
                        if json.loads(json.dumps(generated[key])) != config[key]:
                            raise ValueError(f"Screened structure geometry changed: {key} at {d}/{seed}")
                    record_config(args.output, name, config)
                    groups = structure_boxes(config)
                    vision = endpoint_pairs(arena, config, groups, args.output / f"{name}-sweep.csv")
                    endpoint_source = config.get("endpoint_source", "teacher_outbound")
                    if endpoint_source == "teacher_outbound":
                        teacher = arena.teacher_structure_visibility(
                            config, groups, args.output / f"{name}-teacher.csv")
                    else:
                        # A planned pose does not have a completed teacher
                        # outbound segment. Retain its paired view and control.
                        teacher = dict(teacher_replay="planned_only")
                    controls = {}
                    if not args.structure_ablation_only:
                        arena.command("/gamemode adventure @p")
                        arena.settle()
                        controls = evaluate_control(arena, config, StepTrace(args.output / f"{name}-structure.csv.gz"),
                                                    scan=False, policy=find_structure_policy)
                        arena.command("/gamemode survival @p")
                        arena.settle()
                    result = dict(d=d, seed=seed, world_seed=world, return_limit=config["return_limit"],
                                  endpoint_source=endpoint_source,
                                  **vision, **teacher, **controls)
                    writer.writerow(result)
                    stream.flush()
                    print(result, flush=True)
            finally:
                arena.sim.close()


def score_rows(inputs, frozen):
    expected = set(frozen["formal"])
    if len(expected) != 30 or len(frozen["formal"]) != 30:
        raise ValueError("Structure shortcut acceptance requires 30 distinct frozen formal seeds")
    rows = defaultdict(dict)
    for path in inputs:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                d, seed = int(row["d"]), int(row["seed"])
                if seed in rows[d] or seed not in expected or d not in (30, 60, 100):
                    raise ValueError(f"Duplicate or out-of-cohort structure audit case {d}/{seed}")
                if int(row["world_seed"]) != frozen["world_seed_by_scenario"][str(seed)]:
                    raise ValueError(f"Wrong world seed for structure audit case {d}/{seed}")
                rows[d][seed] = row
    return rows, expected


def missing_cases(inputs, frozen):
    rows, expected = score_rows(inputs, frozen)
    for d in (30, 60, 100):
        measured = {seed for seed, row in rows[d].items() if row["success"] in {"True", "False"}}
        for seed in sorted(expected - measured):
            yield d, seed


def summarize(inputs, frozen):
    rows, expected = score_rows(inputs, frozen)
    aggregate = {}
    for d in (30, 60, 100):
        group = list(rows[d].values())
        observed = [row for row in group if row["success"] in {"True", "False"}]
        authentic = [row for row in observed if row["endpoint_source"] == "teacher_outbound"]
        successes = sum(row["success"] == "True" for row in authentic)
        known = len(authentic)
        if any(row["success"] == "True" and (int(row["deaths"]) or int(row["pose_jumps"]))
               for row in observed):
            raise ValueError("A discontinuity cannot count toward structure shortcut success")
        planned = [int(row["seed"]) for row in group if row["endpoint_source"] != "teacher_outbound"]
        unknown = sorted(expected - {int(row["seed"]) for row in authentic})
        successful_replays = [row for row in group
                              if row["teacher_replay"] == "reached" and row["endpoint_source"] == "teacher_outbound"]
        vision = {}
        for name in STRUCTURES:
            teacher = [float(row[f"{name}_first_dist"]) for row in successful_replays
                       if row[f"{name}_first_dist"]]
            vision[name] = dict(
                start_visible_any=sum(int(row[f"{name}_visible_headings"]) > 0 for row in group),
                start_visible_heading_sum=sum(int(row[f"{name}_visible_headings"]) for row in group),
                teacher_first_visible=len(teacher),
                teacher_first_dist_five_number=np.quantile(teacher, [0, .25, .5, .75, 1]).tolist() if teacher else [],
            )
        total = len(expected)
        complete = len(group) == total == 30
        upper = successes + len(unknown)
        threshold = math.ceil(.10 * total)
        failed = successes >= threshold
        passed = complete and upper < threshold
        aggregate[d] = dict(
            complete=complete, expected=total, vision_cases=len(group),
            planned_endpoint_seeds=sorted(planned),
            planned_endpoint_successes=sorted(int(row["seed"]) for row in observed
                                               if row["endpoint_source"] != "teacher_outbound"
                                               and row["success"] == "True"),
            teacher_replay_success=len(successful_replays),
            shortcut_cases=len(observed), shortcut_known_cases=known, shortcut_unknown_seeds=unknown,
            shortcut_success=successes, shortcut_rate=successes / known if known else None,
            shortcut_ci95=wilson(successes, known) if known else None,
            shortcut_success_upper=upper, shortcut_rate_lower=successes / total,
            shortcut_rate_upper=upper / total,
            threshold_decidable=failed or passed, threshold_exceeded=failed,
            passed=passed, structures=vision,
        )
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--missing-cases", action="store_true",
                        help="Print unscored distance and seed pairs instead of writing a summary")
    args = parser.parse_args()
    frozen = json.loads(args.seeds_file.read_text())
    if args.missing_cases:
        if args.output:
            parser.error("--missing-cases prints the selected pairs and takes no output path")
        for distance, seed in missing_cases(args.inputs, frozen):
            print(distance, seed)
        return
    if not args.output:
        parser.error("An output path is required for a summary")
    result = summarize(args.inputs, frozen)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

"""Pilot matched T1 scenes with two possible visual goals and waypoint control."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import deque
from itertools import product
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from t1_collect import ProtectedCaptureArena
from t1_occlusion import set_candidate_render_distance
from t1_pose_replay import return_by_pose
from t1_respawn import record_config
from t1_safety import StepTrace
from t1_shortcut_audit import gold_target
from t1_v4 import DISTANCES, bearing, horizontal


PILOT_SEED = 2083010000
PILOT_WORLD = 21430100
SPLITS = ("pilot", "train", "validation", "test")
FIELDS = ("seed", "world_seed", "distance", "goal_slot", "success", "steps", "return_limit",
          "min_dist", "final_dist", "deaths", "pose_jump", "pose_jumps", "end_reason",
          "spawn_distance", "cue_pixels_left", "cue_pixels_right", "identical_headings", "error")


def frame_hash(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def configure_rendering(output):
    set_candidate_render_distance(output)
    options = output / "minestudio" / "runtime" / "options.txt"
    settings = {"renderClouds": "false", "particles": "2"}
    entries = [line for line in options.read_text().splitlines()
               if line.partition(":")[0] not in settings]
    options.write_text("\n".join([*entries, *(f"{key}:{value}" for key,value in settings.items())])+"\n")


def ray_intersects(origin, target, box):
    start, end = 0.0, 1.0
    for value, limit, lower, upper in zip(origin, target, box[:3], box[3:], strict=True):
        delta = limit-value
        if abs(delta) < 1e-9:
            if not lower <= value <= upper+1:
                return False
            continue
        near, far = sorted(((lower-value)/delta, (upper+1-value)/delta))
        start, end = max(start, near), min(end, far)
        if start >= end:
            return False
    return start < 1 and end > 0


def candidate_occlusion(layout, y):
    pose = layout["return_pose"]
    origin = (pose["x"], pose["y"]+1.62, pose["z"])
    result = []
    for x,z in layout["candidate_positions"]:
        points = product((math.floor(x), x, math.floor(x)+1), (y, y+1, y+2, y+3),
                         (math.floor(z), z, math.floor(z)+1))
        result.append(all(any(ray_intersects(origin, point, wall) for wall in layout["occlusion_boxes"])
                          for point in points))
    return result


class EnclosureArena(ProtectedCaptureArena):
    memory_frames = None
    door_jamb = 2
    spawn_offset = -13

    def capture_frame(self):
        import cv2

        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(self.info["pov"], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise RuntimeError("Dual-enclosure observation could not be encoded")
        pose = [float(self.info["player_pos"][key]) for key in ("x", "z", "yaw")]
        self.memory_frames.append((encoded.tobytes(), pose))
        return len(self.memory_frames)-1

    def teacher_step(self, yaw_delta=0.0, forward=False, jump=False, record=False):
        if record and self.memory_frames is not None and len(self.outbound) % 8 == 0:
            self.capture_frame()
        return super().teacher_step(yaw_delta, forward, jump, record)

    def layout(self, seed, distance):
        rng = random.Random(seed)
        self.direction = rng.choice(((0, 1), (-1, 0), (0, -1), (1, 0)))
        self.landmark = self.x + .5, self.z + .5
        spacing = min(12, int(.6*distance))
        endpoint = self.project(0, math.sqrt(distance**2-spacing**2))
        spawn = self.project(0, self.spawn_offset)
        order = rng.sample([-spacing, spacing], 2)
        centers = [self.project(slot, 0) for slot in (-spacing, spacing)]
        yaw = rng.uniform(-180, 180)
        return dict(protocol="t1_dual_enclosure", scenario_seed=seed, world_seed=self.world_seed,
                    distance=distance, target_spacing=spacing, candidate_positions=centers,
                    arena_origin=[self.x, self.y, self.z], spawn=spawn, endpoint=endpoint,
                    outbound_room_order=order, corridor=self.project(0, -6),
                    return_pose=dict(x=endpoint[0], y=self.y, z=endpoint[1], yaw=yaw, pitch=0.0),
                    **self.safety_config())

    def panel(self, start, end, low, high, material):
        ax, az = map(math.floor, self.project(*start))
        bx, bz = map(math.floor, self.project(*end))
        box = min(ax, bx), self.y+low, min(az, bz), max(ax, bx), self.y+high, max(az, bz)
        self.fill(*box, material)
        self.scene_fills.append(box)
        return box

    def establish(self, layout, goal_slot):
        self.landmark = self.x + .5, self.z + .5
        self.run_commands(self.protection_commands)
        for box in self.scene_fills:
            self.fill(*box, "air")
        self.scene_fills.clear()
        spacing = layout["target_spacing"]
        checks = []
        occlusion_boxes = []
        for slot, center in enumerate((-spacing, spacing)):
            jamb = self.door_jamb
            for left, right in (((center-3,-3),(center-3,3)), ((center+3,-3),(center+3,3)),
                                ((center-3,3),(center+3,3)), ((center-3,-3),(center-jamb,-3)),
                                ((center+jamb,-3),(center+3,-3))):
                occlusion_boxes.append(self.panel(left, right, 0, 4, "cobblestone"))
                point = self.project(*left)
                checks.append((math.floor(point[0]), self.y+4, math.floor(point[1]), "cobblestone"))
            occlusion_boxes.append(self.panel((center-3,-3), (center+3,3), 5, 5, "cobblestone"))
            self.panel((center,-2), (center,-2), 5, 5, "sea_lantern")
            material = "gold_block" if slot == goal_slot else "lapis_block"
            self.panel((center,0), (center,0), 0, 2, material)
            position = self.project(center, 0)
            checks.append((math.floor(position[0]), self.y+1, math.floor(position[1]), material))
        self.check_blocks(checks)
        first = self.spawn_heading_target(layout)
        start = dict(x=layout["spawn"][0], y=self.y, z=layout["spawn"][1],
                     yaw=bearing(layout["spawn"], first), pitch=0.0)
        self.restore_return_pose({"return_pose": start})
        # The geometry is generated around the platform center. Only the scorer needs the gold position.
        target = layout["candidate_positions"][goal_slot]
        if math.dist(horizontal(self.info), target) < 10:
            raise RuntimeError("The dual-enclosure spawn is too close to a candidate")
        layout.update(goal_slot=goal_slot, landmark=target, walls=self.scene_fills.copy(),
                      occlusion_boxes=occlusion_boxes)
        layout["candidate_occlusion"] = candidate_occlusion(layout, self.y)
        if not all(layout["candidate_occlusion"]):
            raise RuntimeError("A candidate has an unoccluded ray from the paired endpoint")

    def spawn_heading_target(self, layout):
        return self.project(layout["outbound_room_order"][0], -6)

    def traverse_outbound(self, layout):
        cue_pixels = {}
        anchors = {}
        for center in layout["outbound_room_order"]:
            for point in (self.project(center, -6), self.project(center, -2.4)):
                _, reached, _ = self.drive(point, .35, 350, record=True)
                if not reached:
                    raise RuntimeError(f"Outbound teacher missed room waypoint {point}")
            self.face(bearing(horizontal(self.info), self.project(center, 0)), record=True)
            cue_pixels[center] = gold_target(self.info["pov"])[0]
            if self.memory_frames is not None:
                anchors[center] = self.capture_frame()
            elif self.capture is not None:
                anchors[center] = len(self.capture)
            for _ in range(12):
                self.teacher_step(record=True)
            _, reached, _ = self.drive(self.project(center, -6), .35, 350, record=True)
            if not reached:
                raise RuntimeError("Outbound teacher could not leave an enclosure")
        for point in (layout["corridor"], layout["endpoint"]):
            _, reached, _ = self.drive(point, .35, 850, record=True)
            if not reached:
                raise RuntimeError(f"Outbound teacher missed corridor waypoint {point}")
        self.face(layout["return_pose"]["yaw"], record=True)
        layout["room_frame_indices"] = [anchors[slot] for slot in sorted(anchors)]
        return [cue_pixels[slot] for slot in sorted(cue_pixels)]

    def return_targets(self, layout, slot):
        offset = (-1, 1)[slot] * layout["target_spacing"]
        return [layout["corridor"], self.project(offset, -6), self.project(offset, -2.2)]

    def panorama(self, config):
        self.restore_return_pose(config)
        self.settle(12)
        if self.memory_frames is not None:
            config["history_frames"] = len(self.memory_frames)
            for _ in range(8):
                self.capture_frame()
                self.tick()
        frames = []
        trace = StepTrace(None)
        with self.track(trace):
            for _ in range(36):
                frames.append(self.info["pov"].copy())
                self.teacher_step(yaw_delta=10.0)
        return frames


def write_history(output, tag, layout, frames, split):
    path = output / f"{tag}-observations.zip"
    committed = path.with_suffix(".partial")
    metadata = dict(scene=layout, split=split, pose=[pose for _, pose in frames])
    with ZipFile(committed, "x", compression=ZIP_STORED) as archive:
        archive.writestr("episode.json", json.dumps(metadata))
        for index, (image, _) in enumerate(frames):
            archive.writestr(f"{index:05d}.jpg", image)
    committed.replace(path)


def case(arena, seed, distance, output, capture=False, split="pilot", predictions=None, fixtures=None):
    base = arena.layout(seed, distance)
    rows, views, view_hashes = [], [], []
    recordings = []
    for slot in (0, 1):
        tag = f"d{distance}-seed{seed}-goal{slot}"
        layout = dict(base)
        arena.establish(layout, slot)
        arena.history = deque(maxlen=2)
        arena.outbound, arena.path_len_out = [], 0.0
        arena.memory_frames = [] if capture else None
        trace = StepTrace(output / f"{tag}-outbound.csv.gz")
        try:
            with arena.track(trace):
                pixel_counts = arena.traverse_outbound(layout)
            views.append(arena.panorama(layout))
            view_hashes.append([frame_hash(frame) for frame in views[-1]])
            frames = arena.memory_frames
        finally:
            arena.memory_frames = None
        routes = [arena.return_targets(layout, candidate) for candidate in (0, 1)]
        targets = routes[slot]
        # A single waypoint-tracking controller is shared by all heads. No target coordinates enter its input
        # beyond the selected waypoint sequence.
        layout["return_limit"] = round(2 * (sum(math.dist(a, b) for a, b in
                                      zip([layout["endpoint"], *targets], targets))/0.16 + 60) + 200)
        layout.update(teacher_targets=targets, path_len_out=arena.path_len_out,
                      cue_gold_pixels=pixel_counts, endpoint_frame_sha256=view_hashes[-1])
        if fixtures is not None:
            expected = json.loads((fixtures / f"{tag}.json").read_text())
            keys = ("protocol", "scenario_seed", "world_seed", "goal_slot", "candidate_positions",
                    "walls", "spawn", "outbound_room_order", "return_pose", "return_limit", "teacher_targets")
            if any(json.loads(json.dumps(layout[key])) != expected[key] for key in keys):
                raise RuntimeError(f"Frozen paired fixture changed for {tag}")
        record_config(output, tag, layout)
        if capture:
            recordings.append((tag, layout, frames))
        arena.landmark = layout["landmark"]
        policies = {"oracle": slot} if predictions is None else predictions[f"{seed}:{slot}"]
        for name, selected in policies.items():
            if type(selected) is not int or selected not in (0, 1):
                raise ValueError(f"Invalid waypoint branch from {name}")
            returned = StepTrace(output / f"{tag}-{name}.csv.gz")
            result = return_by_pose(arena, layout, routes[selected], returned, terminal_tolerance=0.1)
            row = [seed, arena.world_seed, distance, slot, *result,
                   math.dist(layout["spawn"], layout["landmark"]), *pixel_counts]
            rows.append([*row, name, selected] if predictions is not None else row)
    identical = sum(left == right for left, right in zip(*view_hashes, strict=True))
    differences = []
    for left, right in zip(*views, strict=True):
        delta = np.abs(left.astype(np.int16)-right.astype(np.int16)).max(axis=-1)
        differences.append(dict(changed_pixels=int((delta > 0).sum()), strong_pixels=int((delta > 24).sum()),
                                maximum=int(delta.max()), mean=float(delta.mean())))
    record_config(output, f"d{distance}-seed{seed}-endpoint-audit",
                  dict(identical_headings=identical, differences=differences,
                       candidate_occlusion=layout["candidate_occlusion"]))
    if capture:
        # Both histories receive the same real, stationary endpoint sequence to remove renderer differences.
        source_slot = (seed // 1000) % 2
        shared = recordings[source_slot][2][-8:]
        hashes = [hashlib.sha256(image).hexdigest() for image, _ in shared]
        for tag, layout, frames in recordings:
            layout.update(endpoint_window_source_goal=source_slot, shared_endpoint_window_sha256=hashes)
            record_config(output, tag, layout)
            write_history(output, tag, layout, [*frames[:-8], *shared], split)
    return [(*row[:16], identical, "", *row[16:]) for row in rows]


def run(args):
    configure_rendering(args.output)
    with (args.output / "oracle.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        predictions = json.loads(args.predictions.read_text()) if args.predictions else None
        writer.writerow((*FIELDS, "predictor", "predicted_slot") if predictions is not None else FIELDS)
        split = SPLITS.index(args.split)
        for group in range(args.pairs)[args.worker::args.workers]:
            arena = EnclosureArena(PILOT_WORLD+split*1000+group, effects=True)
            try:
                for distance in args.distances:
                    seed = PILOT_SEED+100000*split+1000*group+distance
                    try:
                        fixtures = (args.fixtures / f"worker-{args.worker:03d}") if args.fixtures else None
                        writer.writerows(case(arena, seed, distance, args.output, args.capture, args.split,
                                              predictions, fixtures))
                    except RuntimeError as error:
                        writer.writerow((seed, arena.world_seed, distance, "", *([""]*13), str(error)))
                    stream.flush()
            finally:
                arena.sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=DISTANCES)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--split", choices=SPLITS, default="pilot")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers <= 32 or args.pairs < 1:
        parser.error("Invalid CPU shard or paired case count")
    if bool(args.predictions) != bool(args.fixtures):
        parser.error("Saved waypoint predictions require their frozen scene fixtures")
    run(args)

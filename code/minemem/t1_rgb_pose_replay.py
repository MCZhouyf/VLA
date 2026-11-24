"""Close the loop with RGB goal retrieval and reverse-route pose memory."""

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from t1_collect import RevisedCaptureArena
from t1_occlusion import set_candidate_render_distance
from t1_pose_replay import FIELDS, recorded_route, return_by_pose, verify_scene
from t1_respawn import record_config
from t1_rgb_anchor_audit import held_out_rows, pick_anchors
from t1_safety import StepTrace
from t1_storage import retry_read


def replay(arena, archive, metadata, output, stride):
    scene = metadata["scene"]
    seed, distance = scene["scenario_seed"], scene["distance"]
    tag = f"d{distance}-seed{seed}"
    config = arena.prepare_episode(seed, distance)
    verify_scene(config, scene)
    config.update(return_pose=scene["return_pose"], return_limit=scene["return_limit"],
                  rgb_anchor_stride=stride)
    choices, _ = pick_anchors(archive, metadata, stride)
    config["rgb_anchor_choices"] = choices
    record_config(output, tag, config)
    arena.command("/gamemode adventure @p")
    arena.settle()
    outbound = metadata["outbound_frames"]
    try:
        for name, (observation, anchor) in choices.items():
            if anchor is None:
                yield dict(d=distance, seed=seed, control=name, world_seed=scene["world_seed"],
                           success=False, steps=0, return_limit=scene["return_limit"], end_reason="no_rgb_match")
                continue
            poses = [tuple(pose[:2]) for pose in metadata["pose"][anchor:outbound]]
            start, route = recorded_route(poses)
            route = route or [start]
            trace = StepTrace(output / f"{tag}-{name}.csv.gz")
            result = return_by_pose(arena, config, route, trace, terminal_tolerance=0.1)
            row = (distance, seed, scene["world_seed"], name, *result, len(route), "")
            yield dict(zip(FIELDS, row, strict=True), chosen_observation=observation, chosen_pose=anchor)
    finally:
        arena.command("/gamemode survival @p")


def run(args):
    index = retry_read(args.dataset.read_bytes)
    set_candidate_render_distance(args.output)
    with (args.output / "rgb-pose-replay.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FIELDS, "chosen_observation", "chosen_pose"))
        writer.writeheader()
        for row in held_out_rows(index)[args.worker::args.workers]:
            path = args.archive_cache / Path(row["path"]).name
            with retry_read(ZipFile, path) as archive:
                metadata = json.loads(retry_read(archive.read, "episode.json"))
                scene = metadata["scene"]
                if scene["scenario_seed"] != row["scene_seed"] or scene["world_seed"] != row["world_seed"]:
                    raise ValueError(f"Unexpected archived scenario {path}")
                arena = RevisedCaptureArena(scene["world_seed"], scene["optional_effects"])
                try:
                    for result in replay(arena, archive, metadata, args.output, args.stride):
                        writer.writerow(result)
                        stream.flush()
                finally:
                    arena.sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for name in ("dataset", "archive-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("worker", "workers"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.stride < 1 or not 0 <= args.worker < args.workers <= 32:
        parser.error("Invalid replay shard or history stride")
    run(args)

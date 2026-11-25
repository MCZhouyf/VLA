"""Verify captured episodes and select balanced, split-disjoint frame budgets."""

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from t1_collect import HELD_OUT_QUOTAS, SPLITS, TRAIN_QUOTAS
from t1_occlusion import SCREENED_PROTOCOL
from t1_storage import trajectory_fingerprint
from t1_v4 import DISTANCES, PROTOCOL


SMOKE_QUOTAS = {**dict.fromkeys(DISTANCES, 1050), "visible": 800}


def read_episodes(roots, screened=False, protocol=None):
    records = []
    seed_ids = set()
    fingerprints = set()
    for root in roots:
        for path in sorted(root.glob("worker-*/episodes.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                seed = int(row["scene_seed"])
                if seed in seed_ids:
                    raise ValueError(f"Duplicate scenario seed: {seed}")
                seed_ids.add(seed)
                archive_path = path.parent / row["path"]
                with ZipFile(archive_path) as archive:
                    meta = json.loads(archive.read("episode.json"))
                    scene = meta["scene"]
                    screened_t1 = screened and row["group"] in DISTANCES
                    expected = protocol or (SCREENED_PROTOCOL if screened_t1 else PROTOCOL)
                    if scene["protocol"] != expected or (
                        screened_t1 and (scene.get("screened_pillar") is not True
                                         or scene.get("render_distance_chunks") != 4)
                    ):
                        raise ValueError(f"Unexpected scene protocol for {archive_path}")
                    assert meta["scene"]["scenario_seed"] == seed
                    assert meta["scene"]["world_seed"] == row["world_seed"]
                    assert meta["group"] == row["group"] and meta["split"] == row["split"]
                    assert meta["outbound_frames"] == row["outbound_frames"]
                    assert len(meta["pose"]) == len(meta["action"]) == row["frames"]
                    assert row["outbound_frames"] + row["return_frames"] == row["frames"]
                    assert row["return_frames"] > 0
                    images = [member for member in archive.infolist()
                              if member.filename.endswith(".jpg")]
                    assert [member.filename for member in images] == [
                        f"{i:05d}.jpg" for i in range(row["frames"])]
                    if "executed_action" in meta:
                        assert len(meta["executed_action"]) == len(meta["perturbation_applied"]) == row["frames"]
                    digest = trajectory_fingerprint(images, meta)
                if digest in fingerprints:
                    raise ValueError(f"Duplicate image and action sequence: {archive_path}")
                fingerprints.add(digest)
                records.append({**row, "path": str(archive_path.resolve()), "fingerprint": digest})
    return sorted(records, key=lambda row: (row["scene_seed"] % 1000, row["scene_seed"]))


def select(records, split, quota_by_group):
    selected = []
    for group, target in quota_by_group.items():
        remaining = target
        for row in records:
            if row["split"] != split or row["group"] != group:
                continue
            used = min(remaining, row["frames"])
            selected.append({**row, "selected_frames": used,
                             "selected_labels": max(0, used - row["outbound_frames"])})
            remaining -= used
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f"{split}/{group} is missing {remaining} frames")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--screened", action="store_true",
                        help="Use screened T1 trajectories alongside the visible-navigation cohort")
    args = parser.parse_args()
    records = read_episodes(args.roots, screened=args.screened)
    frozen = json.loads(args.frozen.read_text())
    benchmark = set(frozen["pilot"]) | set(frozen["formal"])
    if benchmark & {row["scene_seed"] for row in records}:
        raise ValueError("Data collection overlaps with frozen evaluation scenarios")
    world_sets = [{r["world_seed"] for r in records if r["split"] == split} for split in SPLITS]
    frozen_worlds = {frozen["world_seed_by_scenario"][str(seed)] for seed in benchmark}
    if frozen_worlds & set().union(*world_sets):
        raise ValueError("Data collection overlaps with frozen evaluation worlds")
    if sum(map(len, world_sets)) != len(set().union(*world_sets)):
        raise ValueError("Data collection world seeds overlap between splits")
    if any(row["split"] not in SPLITS for row in records):
        raise ValueError("Unexpected dataset split")
    selected = select(records, "train", SMOKE_QUOTAS if args.smoke else TRAIN_QUOTAS)
    if not args.smoke:
        for split in SPLITS[1:]:
            selected.extend(select(records, split, HELD_OUT_QUOTAS))
    totals = Counter()
    labels = Counter()
    available = Counter()
    for row in records:
        available[f"{row['split']}/{row['group']}"] += row["frames"]
    for row in selected:
        key = f"{row['split']}/{row['group']}"
        totals[key] += row["selected_frames"]
        labels[key] += row["selected_labels"]
    if any(number <= 0 for number in labels.values()):
        raise ValueError("A split and group has no return action labels")
    summary = dict(episodes=len(records), selected_episodes=len(selected),
                   captured_frames=dict(available), selected_frames=dict(totals),
                   supervised_frames=dict(labels), duplicate_seeds=0, duplicate_sequences=0,
                   frozen_overlap=0, frozen_world_overlap=0, split_world_overlap=0)
    if args.screened:
        summary["scene_protocols"] = {"t1": SCREENED_PROTOCOL, "visible_navigation": PROTOCOL}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "dataset.jsonl").open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

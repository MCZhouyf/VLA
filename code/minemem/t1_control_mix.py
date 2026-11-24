"""Freeze a balanced 3,000-step continuation index with original and DART labels."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from t1_collect import DART_QUOTAS, NAV_WORLD_SEED, navigation_cases
from t1_data_index import read_episodes, select


LABELS_PER_GROUP = 1200


def select_labels(records, source):
    selected = []
    ordered = list(records)
    random.Random(f"t1-control-mix:{source}").shuffle(ordered)
    for group in DART_QUOTAS:
        remaining = LABELS_PER_GROUP
        for row in ordered:
            if row["split"] != "train" or row["group"] != group or not row["selected_labels"]:
                continue
            use = min(remaining, row["selected_labels"])
            selected.append({**row, "selected_frames": row["outbound_frames"] + use,
                             "selected_labels": use, "control_training_source": source})
            remaining -= use
            if not remaining:
                break
        if remaining:
            raise ValueError(f"{source}/{group} has {remaining} fewer labels than required")
    return selected


def verify_dart(records):
    perturbed, changed, total = Counter(), Counter(), Counter()
    for row in records:
        with ZipFile(row["path"]) as archive:
            meta = json.loads(archive.read("episode.json"))
        actions = meta["action"]
        executed = meta["executed_action"]
        mask = meta["perturbation_applied"]
        start, stop = row["outbound_frames"], row.get("selected_frames", row["frames"])
        if not meta["scene"].get("dart") or any(mask[:start]) or any(
            label != used and not noisy for label, used, noisy in zip(actions, executed, mask, strict=True)
        ):
            raise ValueError(f"Invalid DART trajectory {row['scene_seed']}")
        group = str(row["group"])
        total[group] += max(0, stop - start)
        perturbed[group] += sum(mask[start:stop])
        changed[group] += sum(a != b for a, b in zip(actions[start:stop], executed[start:stop], strict=True))
    return dict(return_labels=total, perturbed_labels=perturbed, correction_differs_from_executed=changed)


def line_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(args):
    original = line_records(args.base_index)
    old = [row for path in args.base_root.glob("*/worker-*/episodes.jsonl") for row in line_records(path)]
    if not old:
        raise ValueError("The original raw episode records are needed for the disjointness check")
    dart = read_episodes(args.dart_root, screened=True)
    frozen = json.loads(args.frozen.read_text())
    benchmark = set(frozen["pilot"]) | set(frozen["formal"])
    frozen_worlds = {frozen["world_seed_by_scenario"][str(seed)] for seed in benchmark} | {NAV_WORLD_SEED}
    frozen_scenes = benchmark | {case[0] for offset in (False, True) for case in navigation_cases(offset)}
    if ({row["scene_seed"] for row in dart} & (frozen_scenes | {row["scene_seed"] for row in old})
            or {row["world_seed"] for row in dart} & (frozen_worlds | {row["world_seed"] for row in old})
            or {row["fingerprint"] for row in dart} & {row["fingerprint"] for row in original}
            or any(row["split"] != "train" for row in dart)):
        raise ValueError("DART data overlaps with an existing split or benchmark")
    selected_dart = select(dart, "train", DART_QUOTAS)
    mixed = select_labels(original, "original") + select_labels(selected_dart, "dart")
    mixed += [row for row in original if row["split"] != "train"]
    stats = verify_dart(selected_dart)
    if any(stats["correction_differs_from_executed"].get(str(group), 0) == 0 for group in DART_QUOTAS):
        raise ValueError("Every DART group must include recorded corrections")
    counts = Counter()
    for row in mixed:
        if row["split"] == "train":
            counts[f'{row["control_training_source"]}/{row["group"]}'] += row["selected_labels"]
    if any(n != LABELS_PER_GROUP for n in counts.values()) or sum(counts.values()) != 12000:
        raise RuntimeError("The continuation schedule does not contain a balanced 12,000 labels")
    args.output.mkdir(parents=True, exist_ok=False)
    index = args.output / "dataset.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in mixed))
    (args.output / "dart-index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected_dart))
    summary = dict(
        original_index_sha256=hashlib.sha256(args.base_index.read_bytes()).hexdigest(),
        mixed_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        original_raw_episodes=len(old), dart_raw_episodes=len(dart), dart_selected_episodes=len(selected_dart),
        dart_raw_frames=sum(row["frames"] for row in dart),
        dart_selected_frames=dict(DART_QUOTAS), **stats, mixed_labels=counts,
        mixed_train_labels=sum(counts.values()), optimizer_updates=sum(counts.values()) // 4,
        dart_scene_seed_range=[min(row["scene_seed"] for row in dart), max(row["scene_seed"] for row in dart)],
        dart_world_seeds=sorted({row["world_seed"] for row in dart}),
        scene_overlap=0, world_overlap=0, frozen_overlap=0,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--dart-root", type=Path, action="append", required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()

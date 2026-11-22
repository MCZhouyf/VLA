"""Train one T1 variant on a shared, frame-budgeted episode index."""

import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path
from time import monotonic
from zipfile import ZipFile

import torch
from PIL import Image

from t1_model import HISTORY, T1Policy


def load_images(archive, indices, wrapper, resolution):
    images = []
    for index in indices:
        with Image.open(io.BytesIO(archive.read(f"{index:05d}.jpg"))) as image:
            frame = image.convert("RGB")
        if resolution:
            frame = frame.resize(resolution, Image.Resampling.BILINEAR)
        images.append(wrapper.create_image_input(frame))
    return images


def case_examples(policy, row, epochs_seed, resolution):
    with ZipFile(row["path"]) as archive:
        meta = json.loads(archive.read("episode.json"))
        start = row["outbound_frames"]
        stop = row["selected_frames"]
        if start >= stop:
            return
        features = None
        if policy.memory is not None:
            features = torch.cat([
                policy.pooled_images(load_images(archive, range(begin, min(begin + 8, stop - 1)),
                                                 policy.images, resolution))
                for begin in range(0, stop - 1, 8)
            ])
        indices = list(range(start, stop))
        random.Random(epochs_seed + row["scene_seed"]).shuffle(indices)
        for action_index in indices:
            first = max(0, action_index - policy.history)
            frames = load_images(archive, range(first, action_index + 1), policy.images, resolution)
            actions = meta["action"][first:action_index]
            if len(actions) < policy.history:
                missing = policy.history - len(actions)
                frames = [frames[0]] * missing + frames
                actions = [policy.actions.null_token()] * missing + actions
            yield (meta["instruction"], frames, actions, meta["action"][action_index],
                   features[:action_index] if features is not None else None,
                   meta["pose"][:action_index + 1] if features is not None else None)


def train(args):
    started = monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset_bytes = args.dataset.read_bytes()
    dataset = [json.loads(line) for line in dataset_bytes.splitlines()]
    dataset = [row for row in dataset if row["split"] == "train" and row["selected_labels"] > 0]
    if not dataset:
        raise ValueError("No return actions in the specified training index")
    torch.manual_seed(args.seed)
    policy = T1Policy(args.checkpoint, args.variant, quantized=args.quantized, training=True)
    policy.train()
    parameters = [param for param in policy.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    examples_per_epoch = sum(row["selected_labels"] for row in dataset)
    total_examples = examples_per_epoch * args.epochs
    settings = dict(
        variant=args.variant, history=policy.history, quantized=args.quantized,
        checkpoint=str(args.checkpoint), dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
        resolution=args.resolution, seed=args.seed, epochs=args.epochs,
        examples_per_epoch=examples_per_epoch, accumulation=args.accumulation,
        trainable_parameters=sum(p.numel() for p in parameters), learning_rate=args.learning_rate,
        budget_seconds=args.budget_seconds,
    )
    (args.output / "training_config.json").write_text(json.dumps(settings, indent=2) + "\n")

    updates = samples = group_count = 0
    group_loss = 0.0
    first = last = None
    stopped = False
    with (args.output / "loss.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("update", "examples", "loss", "elapsed_seconds", "peak_allocated_mib"))
        for epoch in range(args.epochs):
            order = dataset.copy()
            random.Random(args.seed + epoch).shuffle(order)
            for row in order:
                for example in case_examples(policy, row, args.seed + epoch, args.resolution):
                    count = min(args.accumulation, total_examples - samples + group_count)
                    loss = policy.loss(*example)
                    measured = float(loss.detach())
                    if not math.isfinite(measured):
                        raise RuntimeError(f"Nonfinite action loss at example {samples + 1}")
                    (loss / count).backward()
                    group_loss += measured
                    group_count += 1
                    samples += 1
                    if group_count < count:
                        continue
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    last = group_loss / group_count
                    if first is None:
                        first = last
                    elapsed = monotonic() - started
                    writer.writerow((updates, samples, last, elapsed,
                                     torch.cuda.max_memory_allocated() / 2**20))
                    stream.flush()
                    if updates % 10 == 0:
                        print(f"update={updates} samples={samples} loss={last:.5f}", flush=True)
                    group_count = 0
                    group_loss = 0.0
                    if args.budget_seconds and elapsed >= args.budget_seconds:
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break
    if not updates:
        raise RuntimeError("No optimizer updates completed")
    policy.model.save_pretrained(args.output / "adapter", safe_serialization=True)
    if policy.memory is not None:
        from safetensors.torch import save_file

        save_file({name: value.detach().cpu().contiguous()
                   for name, value in policy.memory.state_dict().items()},
                  args.output / "spatial.safetensors")
    result = dict(updates=updates, supervised_examples=samples, elapsed_seconds=monotonic() - started,
                  first_update_loss=first, last_update_loss=last,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  stopped_at_budget=stopped)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(HISTORY), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--budget-seconds", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--quantized", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.accumulation <= 0 or args.budget_seconds < 0:
        parser.error("Invalid epochs, accumulation, or time budget")
    train(args)


if __name__ == "__main__":
    main()

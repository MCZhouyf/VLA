"""Train one T1 variant on a shared, frame-budgeted episode index."""

import argparse
import csv
import hashlib
import io
import json
import math
import random
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from time import monotonic
from zipfile import ZipFile

import torch
from PIL import Image
from torch.nn import functional as F

from t1_model import HISTORY, T1Policy
from t1_storage import retry_read
from t1_train_state import checkpoints, load_checkpoint, save_checkpoint, save_weights, trim_log


def load_images(archive, indices, wrapper, resolution):
    images = []
    for index in indices:
        with Image.open(io.BytesIO(retry_read(archive.read, f"{index:05d}.jpg"))) as image:
            frame = image.convert("RGB")
        if resolution:
            frame = frame.resize(resolution, Image.Resampling.BILINEAR)
        images.append(wrapper.create_image_input(frame))
    return images


def encode_episode(policy, archive, start, stop, resolution):
    history_start = max(0, start - policy.history)
    has_memory = policy.memory is not None
    frames, pooled = [], []
    first_frame = 0 if has_memory else history_start
    for begin in range(first_frame, stop, 3):
        end = min(begin + 3, stop)
        images = load_images(archive, range(begin, end), policy.images, resolution)
        # Early outbound frames only feed the spatial writer. Pool them before
        # copying to the CPU, where only the recent image tokens are needed.
        storage = policy.device if end <= history_start else "cpu"
        encoded = policy.encode_frames(images, storage=storage)
        if has_memory and begin < stop - 1:
            pooled.append(policy.pooled_images(encoded[:stop - 1 - begin]))
        frames.extend(encoded[max(0, history_start - begin):])
    features = None
    if has_memory:
        features = (torch.cat(pooled) if pooled else
                    torch.empty((0, policy.base.config.hidden_size), dtype=torch.float16))
    return history_start, frames, features


def case_examples(policy, row, epochs_seed, resolution, cache_vision=False,
                  archive_cache=None, validation_frames=None, skip=0):
    source = Path(row["path"])
    if archive_cache is not None:
        source = archive_cache / source.name
    start = row["outbound_frames"]
    stop = row["selected_frames"]
    if start + skip >= stop:
        return
    with retry_read(ZipFile, source) as archive:
        meta = json.loads(retry_read(archive.read, "episode.json"))
        features = None
        if cache_vision:
            encoded = None if validation_frames is None else validation_frames.get(row["path"])
            if encoded is None:
                encoded = encode_episode(policy, archive, start, stop, resolution)
                if validation_frames is not None:
                    validation_frames[row["path"]] = encoded
            offset, cached, features = encoded
        elif policy.memory is not None:
            features = torch.cat([
                policy.pooled_images(load_images(archive, range(begin, min(begin + 8, stop - 1)),
                                                 policy.images, resolution))
                for begin in range(0, stop - 1, 8)
            ])
        indices = list(range(start, stop))
        random.Random(epochs_seed + row["scene_seed"]).shuffle(indices)
        for action_index in indices[skip:]:
            first = max(0, action_index - policy.history)
            frames = (cached[first-offset:action_index+1-offset] if cache_vision else
                      load_images(archive, range(first, action_index + 1), policy.images, resolution))
            actions = meta["action"][first:action_index]
            if len(actions) < policy.history:
                missing = policy.history - len(actions)
                frames = [frames[0]] * missing + frames
                actions = [policy.actions.null_token()] * missing + actions
            yield (meta["instruction"], frames, actions, meta["action"][action_index],
                   features[:action_index] if features is not None else None,
                   meta["pose"][:action_index + 1] if features is not None else None)


VALIDATION_FIELDS = ("update", "frames", "actions", "tokens", "loss",
                     "token_accuracy", "exact_action_accuracy", "seconds", "device", "dtype")


@torch.inference_mode()
def validation_metrics(policy, dataset, seed, resolution, update, *, cache_vision=False,
                       archive_cache=None, validation_frames=None, batch_size=1):
    started = monotonic()
    training = policy.training
    frames = sum(row["selected_frames"] for row in dataset)
    actions = tokens = 0
    correct = torch.zeros((), device=policy.device, dtype=torch.long)
    exact = torch.zeros_like(correct)
    total = torch.zeros((), device=policy.device, dtype=torch.float64)
    policy.eval()
    try:
        examples = (example for row in dataset for example in
                    case_examples(policy, row, seed, resolution, cache_vision,
                                  archive_cache, validation_frames))
        while group := list(islice(examples, batch_size)):
            predictions = (policy.batched_action_logits(group) if len(group) > 1 else
                           [policy.action_logits(*group[0])])
            for logits, targets in predictions:
                total += F.cross_entropy(logits, targets, reduction="sum")
                matched = logits.argmax(dim=-1).eq(targets).sum()
                tokens += targets.numel()
                correct += matched
                actions += 1
                exact += matched == targets.numel()
    finally:
        policy.train(training)
    if actions != sum(row["selected_labels"] for row in dataset) or not tokens or not math.isfinite(total):
        raise RuntimeError("Invalid validation coverage or loss")
    return (update, frames, actions, tokens, float(total)/tokens, int(correct)/tokens, int(exact)/actions,
            monotonic()-started, str(policy.device), str(policy.base.dtype))


def training_examples(policy, dataset, args, cursor):
    initial_epoch, initial_row, initial_offset = cursor
    for epoch in range(initial_epoch, args.epochs):
        order = dataset.copy()
        random.Random(args.seed + epoch).shuffle(order)
        row_start = initial_row if epoch == initial_epoch else 0
        for row_index in range(row_start, len(order)):
            offset = initial_offset if (epoch, row_index) == (initial_epoch, initial_row) else 0
            examples = case_examples(policy, order[row_index], args.seed + epoch, args.resolution,
                                     args.cache_vision, args.archive_cache, skip=offset)
            for example_index, example in enumerate(examples, start=offset):
                yield (epoch, row_index, example_index + 1), example


def microbatch_losses(policy, examples):
    if len(examples) == 1:
        return policy.loss(*examples[0]).unsqueeze(0)
    return torch.stack([F.cross_entropy(logits, targets)
                        for logits, targets in policy.batched_action_logits(examples)])


def train(args):
    started = monotonic()
    snapshots = args.output / "checkpoints"
    if args.resume:
        available = checkpoints(snapshots)
        if not available or (args.output / "summary.json").is_file():
            raise ValueError("Resume requires an unfinished run with a complete checkpoint")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    dataset_bytes = retry_read(args.dataset.read_bytes)
    rows = [json.loads(line) for line in dataset_bytes.splitlines()]
    dataset = [row for row in rows if row["split"] == "train" and row["selected_labels"] > 0]
    validation = [row for row in rows if row["split"] == "validation"] if args.validate_every else []
    if args.validate_every and sum(row["selected_frames"] for row in validation) != 6500:
        raise ValueError("Periodic validation requires the complete 6,500-frame split")
    if not dataset:
        raise ValueError("No return actions in the specified training index")
    torch.manual_seed(args.seed)
    policy = T1Policy(args.checkpoint, args.variant, quantized=args.quantized, training=True)
    policy.train()
    parameters = [param for param in policy.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer.zero_grad(set_to_none=True)
    examples_per_epoch = sum(row["selected_labels"] for row in dataset)
    total_examples = examples_per_epoch * args.epochs
    settings = dict(
        variant=args.variant, history=policy.history, quantized=args.quantized,
        checkpoint=str(args.checkpoint), dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
        resolution=args.resolution, seed=args.seed, epochs=args.epochs,
        examples_per_epoch=examples_per_epoch, accumulation=args.accumulation,
        trainable_parameters=sum(p.numel() for p in parameters), learning_rate=args.learning_rate,
        budget_seconds=args.budget_seconds, validate_every=args.validate_every,
        checkpoint_selection="last_optimizer_step", cache_vision=args.cache_vision,
        validation_batch_size=args.validation_batch_size,
        checkpoint_every=args.checkpoint_every,
        train_microbatch_size=args.train_microbatch_size,
    )
    updates = samples = 0
    first = last = None
    cursor = (0, 0, 0)
    previous_elapsed = 0.0
    stopped = False
    config_path = args.output / "training_config.json"
    loss_path = args.output / "loss.csv"
    val_path = args.output / "val_loss.csv"
    if args.resume:
        saved_settings = json.loads(config_path.read_text())
        saved_settings.setdefault("train_microbatch_size", 1)
        if settings != saved_settings:
            raise ValueError("Resume settings differ from the saved run")
        progress = load_checkpoint(available[-1], policy, optimizer, scheduler)
        updates, samples = progress["updates"], progress["samples"]
        cursor = progress["cursor"]
        first, last, previous_elapsed = progress["first"], progress["last"], progress["elapsed"]
        trim_log(loss_path, updates)
        if validation:
            trim_log(val_path, updates)
        print(f"Resuming update={updates}, examples={samples}, cursor={progress['cursor']}", flush=True)
    else:
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
    validation_frames = {} if validation and args.cache_vision else None
    open_mode = "a" if args.resume else "x"
    val_context = val_path.open(open_mode, newline="") if validation else nullcontext()
    with loss_path.open(open_mode, newline="") as stream, val_context as val_stream:
        writer = csv.writer(stream)
        val_writer = csv.writer(val_stream) if val_stream else None
        if not args.resume:
            writer.writerow(("update", "examples", "loss", "elapsed_seconds", "peak_allocated_mib"))
            stream.flush()
            if val_writer:
                val_writer.writerow(VALIDATION_FIELDS)
                val_stream.flush()
            if args.checkpoint_every:
                save_checkpoint(snapshots, policy, optimizer, scheduler,
                                dict(updates=0, samples=0, cursor=(0, 0, 0), first=None, last=None,
                                     elapsed=monotonic() - started))

        def validate():
            metrics = validation_metrics(policy, validation, args.seed, args.resolution, updates,
                                         cache_vision=args.cache_vision, archive_cache=args.archive_cache,
                                         validation_frames=validation_frames,
                                         batch_size=args.validation_batch_size)
            val_writer.writerow(metrics)
            val_stream.flush()
            print(f"validation update={updates} loss={metrics[4]:.6f} accuracy={metrics[5]:.6f}",
                  flush=True)

        examples = training_examples(policy, dataset, args, cursor)
        while samples < total_examples:
            count = min(args.accumulation, total_examples - samples)
            measured = []
            for begin in range(0, count, args.train_microbatch_size):
                size = min(args.train_microbatch_size, count - begin)
                group = list(islice(examples, size))
                if len(group) != size:
                    raise RuntimeError(f"Training index ended after {samples + begin} supervised examples")
                losses = microbatch_losses(policy, [example for _, example in group])
                micro_losses = losses.detach().tolist()
                if not all(map(math.isfinite, micro_losses)):
                    raise RuntimeError(f"Nonfinite action loss near example {samples + begin + 1}")
                (losses.sum() / count).backward()
                measured.extend(micro_losses)
                cursor = group[-1][0]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            samples += count
            last = sum(measured) / count
            if first is None:
                first = last
            elapsed = previous_elapsed + monotonic() - started
            writer.writerow((updates, samples, last, elapsed, torch.cuda.max_memory_allocated() / 2**20))
            stream.flush()
            if updates % 10 == 0:
                print(f"update={updates} samples={samples} loss={last:.5f}", flush=True)
            if val_writer and updates % args.validate_every == 0:
                validate()
            if args.checkpoint_every and updates % args.checkpoint_every == 0:
                save_checkpoint(snapshots, policy, optimizer, scheduler,
                                dict(updates=updates, samples=samples, cursor=cursor,
                                     first=first, last=last, elapsed=previous_elapsed + monotonic() - started))
            if args.budget_seconds and elapsed >= args.budget_seconds:
                stopped = True
                break
        if not stopped and next(examples, None) is not None:
            raise RuntimeError("Training index produced more supervised examples than declared")
        if val_writer and updates and updates % args.validate_every:
            validate()
    if not updates:
        raise RuntimeError("No optimizer updates completed")
    if not stopped and samples != total_examples:
        raise RuntimeError(f"Expected {total_examples} supervised examples, processed {samples}")
    save_weights(policy, args.output)
    result = dict(updates=updates, supervised_examples=samples,
                  elapsed_seconds=previous_elapsed + monotonic() - started,
                  first_update_loss=first, last_update_loss=last,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  stopped_at_budget=stopped)
    summary = args.output / ".summary.json.tmp"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    summary.replace(args.output / "summary.json")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(HISTORY), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--train-microbatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--budget-seconds", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-vision", action="store_true")
    parser.add_argument("--archive-cache", type=Path)
    args = parser.parse_args()
    if args.validate_every is None:
        schedule = args.output.parent / "validation_schedule.json"
        args.validate_every = int(json.loads(schedule.read_text())["interval"]) if schedule.is_file() else 0
    if (args.epochs <= 0 or not 0 < args.train_microbatch_size <= args.accumulation or args.budget_seconds < 0
            or args.validate_every < 0 or args.validation_batch_size <= 0 or args.checkpoint_every < 0):
        parser.error("Invalid epochs, accumulation, or time budget")
    train(args)


if __name__ == "__main__":
    main()

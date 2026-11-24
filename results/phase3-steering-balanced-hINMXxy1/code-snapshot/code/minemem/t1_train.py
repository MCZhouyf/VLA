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
        history_actions = meta.get("executed_action", meta["action"])
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
            actions = history_actions[first:action_index]
            if len(actions) < policy.history:
                missing = policy.history - len(actions)
                frames = [frames[0]] * missing + frames
                actions = [policy.actions.null_token()] * missing + actions
            yield (meta["instruction"], frames, actions, meta["action"][action_index],
                   features[:action_index] if features is not None else None,
                   meta["pose"][:action_index + 1] if features is not None else None)


VALIDATION_FIELDS = ("update", "frames", "actions", "tokens", "loss",
                     "token_accuracy", "exact_action_accuracy", "seconds", "device", "dtype",
                     "changed_actions", "changed_action_exact_accuracy")


def mask_history(example, placeholder):
    instruction, frames, actions, target, features, poses = example
    return instruction, frames, [placeholder] * len(actions), target, features, poses


@torch.inference_mode()
def validation_metrics(policy, dataset, seed, resolution, update, *, cache_vision=False,
                       archive_cache=None, validation_frames=None, batch_size=1,
                       use_history_placeholder=False):
    started = monotonic()
    training = policy.training
    frames = sum(row["selected_frames"] for row in dataset)
    actions = tokens = 0
    correct = torch.zeros((), device=policy.device, dtype=torch.long)
    exact = torch.zeros_like(correct)
    exact_changed = torch.zeros_like(correct)
    changed_actions = 0
    total = torch.zeros((), device=policy.device, dtype=torch.float64)
    policy.eval()
    try:
        examples = (example for row in dataset for example in
                    case_examples(policy, row, seed, resolution, cache_vision,
                                  archive_cache, validation_frames))
        while group := list(islice(examples, batch_size)):
            inputs = ([mask_history(example, policy.actions.null_token()) for example in group]
                      if use_history_placeholder else group)
            predictions = (policy.batched_action_logits(inputs) if len(inputs) > 1 else
                           [policy.action_logits(*inputs[0])])
            for example, (logits, targets) in zip(group, predictions, strict=True):
                total += F.cross_entropy(logits, targets, reduction="sum")
                matched = logits.argmax(dim=-1).eq(targets).sum()
                tokens += targets.numel()
                correct += matched
                actions += 1
                exact += matched == targets.numel()
                if example[3] != example[2][-1]:
                    changed_actions += 1
                    exact_changed += matched == targets.numel()
    finally:
        policy.train(training)
    if actions != sum(row["selected_labels"] for row in dataset) or not tokens or not math.isfinite(total):
        raise RuntimeError("Invalid validation coverage or loss")
    return (update, frames, actions, tokens, float(total)/tokens, int(correct)/tokens, int(exact)/actions,
            monotonic()-started, str(policy.device), str(policy.base.dtype), changed_actions,
            int(exact_changed)/changed_actions if changed_actions else "")


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


def source_epoch(policy, rows, args, epoch, skip):
    order = rows.copy()
    random.Random(args.seed + epoch).shuffle(order)
    for row in order:
        if skip >= row["selected_labels"]:
            skip -= row["selected_labels"]
            continue
        yield from case_examples(policy, row, args.seed + epoch, args.resolution,
                                 args.cache_vision, args.archive_cache, skip=skip)
        skip = 0


def balanced_training_examples(policy, sources, per_source, args, samples):
    start_epoch, start_sample = divmod(samples, 2 * per_source)
    if start_sample % 2:
        raise ValueError("A balanced run must resume at a completed source pair")
    for epoch in range(start_epoch, args.epochs):
        skip = start_sample // 2 if epoch == start_epoch else 0
        streams = [source_epoch(policy, rows, args, epoch, skip) for rows in sources]
        for position, pair in enumerate(zip(*streams, strict=True), start=skip):
            for source, example in enumerate(pair):
                yield (epoch, source, 2 * position + source + 1), example


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
    sources = None
    if getattr(args, "balance_sources", False):
        source_names = ("original", "dart")
        if {row.get("control_training_source") for row in dataset} != set(source_names):
            raise ValueError("Balanced sampling requires original and DART training sources")
        sources = [[row for row in dataset if row["control_training_source"] == name]
                   for name in source_names]
        original_count, dart_count = (sum(row["selected_labels"] for row in part)
                                      for part in sources)
        if original_count != dart_count or args.accumulation % 2:
            raise ValueError("Balanced sampling requires equal source counts and even accumulation")
    examples_per_epoch = sum(row["selected_labels"] for row in dataset)
    full_examples = examples_per_epoch * args.epochs
    max_updates = getattr(args, "max_updates", None)
    total_examples = min(full_examples, max_updates * args.accumulation) if max_updates else full_examples
    torch.manual_seed(args.seed)
    initialization = args.initialize_from
    if initialization is not None:
        initial_config = json.loads((initialization / "training_config.json").read_text())
        initial_result = json.loads((initialization / "summary.json").read_text())
        if ((initial_config["variant"], initial_config["quantized"], initial_config["resolution"])
                != (args.variant, args.quantized, args.resolution) or initial_result["stopped_at_budget"]):
            raise ValueError("Initialization checkpoint does not match the new training configuration")
    policy = T1Policy(args.checkpoint, args.variant, quantized=args.quantized, training=True,
                      adapter=initialization / "adapter" if initialization is not None else None)
    if initialization is not None and policy.memory is not None:
        from safetensors.torch import load_file

        policy.memory.load_state_dict(load_file(initialization / "spatial.safetensors"))
    policy.train()
    parameters = [param for param in policy.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer.zero_grad(set_to_none=True)
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
    if initialization is not None:
        settings.update(initialize_from=str(initialization), initialize_update=initial_result["updates"])
    if args.history_dropout:
        settings.update(history_dropout=args.history_dropout, evaluation_history="placeholder")
    if sources is not None:
        settings.update(balance_sources=list(source_names))
    if max_updates:
        settings.update(max_updates=max_updates)
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
        fields = VALIDATION_FIELDS
        if args.resume and val_writer:
            with val_path.open(newline="") as stream:
                fields = tuple(next(csv.reader(stream)))
            if fields not in (VALIDATION_FIELDS, VALIDATION_FIELDS[:-2]):
                raise ValueError("Unexpected validation log fields")
        if not args.resume:
            writer.writerow(("update", "examples", "loss", "elapsed_seconds", "peak_allocated_mib"))
            stream.flush()
            if val_writer:
                val_writer.writerow(fields)
                val_stream.flush()
            if args.checkpoint_every:
                save_checkpoint(snapshots, policy, optimizer, scheduler,
                                dict(updates=0, samples=0, cursor=(0, 0, 0), first=None, last=None,
                                     elapsed=monotonic() - started))

        def validate():
            metrics = validation_metrics(policy, validation, args.seed, args.resolution, updates,
                                         cache_vision=args.cache_vision, archive_cache=args.archive_cache,
                                         validation_frames=validation_frames,
                                         batch_size=args.validation_batch_size,
                                         use_history_placeholder=bool(args.history_dropout))
            val_writer.writerow(metrics[:len(fields)])
            val_stream.flush()
            print(f"validation update={updates} loss={metrics[4]:.6f} accuracy={metrics[5]:.6f}",
                  flush=True)

        examples = (balanced_training_examples(policy, sources, original_count, args, samples)
                    if sources is not None else training_examples(policy, dataset, args, cursor))
        while samples < total_examples:
            count = min(args.accumulation, total_examples - samples)
            measured = []
            for begin in range(0, count, args.train_microbatch_size):
                size = min(args.train_microbatch_size, count - begin)
                group = list(islice(examples, size))
                if len(group) != size:
                    raise RuntimeError(f"Training index ended after {samples + begin} supervised examples")
                inputs = [example for _, example in group]
                if args.history_dropout:
                    inputs = [mask_history(example, policy.actions.null_token())
                              if torch.rand(()).item() < args.history_dropout else example
                              for example in inputs]
                losses = microbatch_losses(policy, inputs)
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
        if not stopped and total_examples == full_examples and next(examples, None) is not None:
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
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--history-dropout", type=float, default=0.0)
    parser.add_argument("--balance-sources", action="store_true",
                        help="Alternate original and DART labels within each optimizer update")
    parser.add_argument("--max-updates", type=int,
                        help="Train on a deterministic prefix of the selected epoch order")
    args = parser.parse_args()
    if args.validate_every is None:
        schedule = args.output.parent / "validation_schedule.json"
        args.validate_every = int(json.loads(schedule.read_text())["interval"]) if schedule.is_file() else 0
    if (args.epochs <= 0 or not 0 < args.train_microbatch_size <= args.accumulation or args.budget_seconds < 0
            or args.validate_every < 0 or args.validation_batch_size <= 0 or args.checkpoint_every < 0
            or not 0 <= args.history_dropout <= 1 or (args.max_updates is not None and args.max_updates <= 0)):
        parser.error("Invalid epochs, accumulation, or time budget")
    train(args)


if __name__ == "__main__":
    main()

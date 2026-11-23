"""Optimizer-boundary snapshots for T1 LoRA training."""

import csv
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def checkpoints(root):
    return sorted(path for path in root.glob("step-*")
                  if path.is_dir() and (path / "state.pt").is_file())


def save_weights(policy, path):
    policy.model.save_pretrained(path / "adapter", safe_serialization=True)
    if policy.memory is not None:
        from safetensors.torch import save_file

        save_file({name: value.detach().cpu().contiguous()
                   for name, value in policy.memory.state_dict().items()},
                  path / "spatial.safetensors")


def save_checkpoint(root, policy, optimizer, scheduler, progress):
    destination = root / f"step-{progress['updates']:08d}"
    root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=root, prefix=".writing-") as temp:
        staged = Path(temp) / destination.name
        staged.mkdir()
        save_weights(policy, staged)
        torch.save(dict(progress=progress, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(policy.device)
                        if policy.device.type == "cuda" else None),
                   staged / "state.pt")
        staged.rename(destination)
    for old in checkpoints(root)[:-2]:
        shutil.rmtree(old)


def load_checkpoint(path, policy, optimizer, scheduler):
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    adapter = load_peft_weights(path / "adapter", device="cpu")
    set_peft_model_state_dict(policy.model, adapter)
    if policy.memory is not None:
        from safetensors.torch import load_file

        policy.memory.load_state_dict(load_file(path / "spatial.safetensors"))
    saved = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["cpu_rng"])
    if saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state(saved["cuda_rng"], device=policy.device)
    return saved["progress"]


def trim_log(path, update):
    with path.open(newline="") as source:
        rows = csv.reader(source)
        kept = [next(rows)]
        for row in rows:
            try:
                if int(row[0]) <= update:
                    kept.append(row)
            except (IndexError, ValueError):
                continue
    temp = path.with_suffix(".resume")
    with temp.open("w", newline="") as stream:
        csv.writer(stream).writerows(kept)
    temp.replace(path)

"""CPU integration check of T1 checkpoint recovery and cached prompt semantics."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from jarvisvla.inference.action_mapping import OneActionTokenizer
from jarvisvla.inference.processor_wrapper import ProcessorWrapper
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2VLConfig, Qwen2VLForConditionalGeneration

import t1_train
import t1_spatial
from t1_model import HISTORY, LORA_TARGETS, T1Policy
from t1_spatial import CELL_BLOCKS, SIDE, SpatialMemory, retained_cells
from t1_storage import retry_read
from t1_v4 import INSTRUCTION, VISIBLE_INSTRUCTION


def tiny_policy(checkpoint, variant, adapter=None, **_):
    policy = T1Policy.__new__(T1Policy)
    torch.nn.Module.__init__(policy)
    policy.history = HISTORY[variant]
    policy.images = ProcessorWrapper(model_name="qwen2_vl")
    policy.actions = OneActionTokenizer(tokenizer_type="qwen2_vl")
    policy.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    public = Qwen2VLConfig.from_pretrained(checkpoint, local_files_only=True)
    config = Qwen2VLConfig(
        vocab_size=public.vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling=dict(type="mrope", mrope_section=[2, 3, 3]),
        vision_start_token_id=public.vision_start_token_id,
        image_token_id=public.image_token_id, video_token_id=public.video_token_id,
        vision_config=dict(depth=1, embed_dim=64, hidden_size=64, num_heads=4),
    )
    config._attn_implementation = "sdpa"
    base = Qwen2VLForConditionalGeneration(config)
    if adapter is not None:
        policy.model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        policy.model = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
        ))
    policy.memory = SpatialMemory(64, 64, width=32) if variant == "spatial" else None
    return policy


def make_dataset(root, label, turn_label=None):
    paths = []
    for case in range(2):
        path = root / f"episode-{case}.zip"
        outbound = 2 + 3 * case
        frames = outbound + 5
        actions = [label] * frames
        if turn_label is not None:
            actions[outbound + 1:outbound + 3] = [turn_label, turn_label]
        with ZipFile(path, "w") as archive:
            rng = np.random.default_rng(case)
            for i in range(frames):
                data = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (84, 112, 3), dtype=np.uint8)).save(data, "JPEG")
                archive.writestr(f"{i:05d}.jpg", data.getvalue())
            archive.writestr("episode.json", json.dumps(dict(
                instruction=VISIBLE_INSTRUCTION, action=actions,
                pose=[[i*0.25, case, -90] for i in range(frames)])))
        paths.append(dict(path=str(path), split="train", scene_seed=case, outbound_frames=outbound,
                          selected_frames=frames, selected_labels=frames-outbound,
                          control_training_source=("original", "dart")[case]))
    dataset = root / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in paths))
    return dataset, paths


def verify_context(policy, rows):
    policy.eval()
    raw = [next(t1_train.case_examples(policy, row, 88, None)) for row in rows]
    cached = [next(t1_train.case_examples(policy, row, 88, None, True)) for row in rows]
    cached[1] = ("Return to the gold block that you saw before.", *cached[1][1:])
    raw[1] = (cached[1][0], *raw[1][1:])
    with torch.inference_mode():
        for a, b in zip(raw, cached, strict=True):
            raw_prompt = policy.prompts(*a[:3])
            cached_prompt = policy.prompts(*b[:3])
            for name in ("input_ids", "image_grid_thw"):
                torch.testing.assert_close(raw_prompt[name], cached_prompt[name], rtol=0, atol=0)
            torch.testing.assert_close(policy.action_logits(*a)[0], policy.action_logits(*b)[0],
                                       rtol=1e-4, atol=1e-5)
        for expected, actual in zip(cached, policy.batched_action_logits(cached), strict=True):
            torch.testing.assert_close(policy.action_logits(*expected)[0], actual[0],
                                       rtol=1e-4, atol=1e-5)
        inputs = [(instruction, frames, acts, features, poses)
                  for instruction, frames, acts, _, features, poses in cached]
        singles = [policy.predict(*case)[1:] for case in inputs]
        batch = [output[1:] for output in policy.predict_batch(inputs)]
        assert singles == batch
    one = t1_train.validation_metrics(policy, rows, 88, None, 1)
    batched = t1_train.validation_metrics(policy, rows, 88, None, 1,
                                          cache_vision=True, validation_frames={}, batch_size=4)
    assert one[:4] == batched[:4]
    np.testing.assert_allclose(one[4:7], batched[4:7], rtol=1e-5, atol=1e-6)
    assert one[10:] == batched[10:]


def verify_microbatch(policy, rows):
    policy.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    cases = [next(t1_train.case_examples(policy, row, 88, None, True)) for row in rows]
    cases[1] = ("Return to the gold block that you saw before.", *cases[1][1:])
    trainable = [param for param in policy.parameters() if param.requires_grad]
    single = torch.stack([policy.loss(*case) for case in cases])
    expected = torch.autograd.grad(single.mean(), trainable, allow_unused=True)
    batch = t1_train.microbatch_losses(policy, cases)
    actual = torch.autograd.grad(batch.mean(), trainable, allow_unused=True)
    torch.testing.assert_close(single, batch, atol=1e-5, rtol=1e-4)
    assert any(grad is not None and torch.count_nonzero(grad) for grad in expected)
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-3)


def verify_memory_math():
    rng = np.random.default_rng(51)
    for length in (1, 2, 29, 64, 177, 512, 1201):
        for stride in (0.2, 4.0, 70.0):
            poses = np.cumsum(rng.normal(0, stride, (length, 3)), axis=0)
            cells = np.floor(poses[:, :2] / CELL_BLOCKS).astype(np.int64)
            origin = cells[0] - SIDE // 2
            origins = []
            for cell in cells:
                origin = np.clip(origin, cell - SIDE + 1, cell)
                origins.append(origin)
            lower = np.maximum.accumulate(np.array(origins)[::-1], axis=0)[::-1]
            upper = np.minimum.accumulate((np.array(origins) + SIDE - 1)[::-1], axis=0)[::-1]
            expected = cells[:-1], origins[-1], ((cells >= lower) & (cells <= upper)).all(1)[:-1]
            for a, b in zip(expected, retained_cells(poses), strict=True):
                np.testing.assert_array_equal(a, b)
    if not hasattr(t1_spatial, "write_weights"):
        return
    for length, slots in ((1, 1), (20, 5), (127, 3), (800, 1), (1024, 90)):
        inverse = torch.from_numpy(rng.integers(slots, size=length))
        gate = torch.from_numpy(rng.uniform(.01, .99, size=length).astype(np.float32)).requires_grad_()
        later = (inverse[:, None] == inverse[None, :]).triu(1).float()
        expected = gate * torch.exp(later @ torch.log1p(-gate))
        actual = t1_spatial.write_weights(gate, inverse, slots)
        torch.testing.assert_close(expected, actual, atol=1e-7, rtol=2e-5)
        weights = torch.linspace(-math.pi, math.pi, steps=length)
        grad_old, = torch.autograd.grad(expected @ weights, gate)
        grad_new, = torch.autograd.grad(actual @ weights, gate)
        torch.testing.assert_close(grad_old, grad_new, atol=1e-5, rtol=2e-4)


def recover(root, args, variant):
    reference = root / f"{variant}-reference"
    interrupted = root / f"{variant}-recover"
    t1_train.train(SimpleNamespace(**args, output=reference, variant=variant, resume=False))
    original = t1_train.save_checkpoint

    def crash_after_update(path, policy, optimizer, scheduler, progress):
        if progress["updates"] == 3:
            raise InterruptedError("Injected failure after the loss log was flushed")
        original(path, policy, optimizer, scheduler, progress)

    with patch.object(t1_train, "save_checkpoint", crash_after_update):
        try:
            t1_train.train(SimpleNamespace(**args, output=interrupted, variant=variant, resume=False))
        except InterruptedError:
            pass
        else:
            raise AssertionError("The expected injected interruption did not occur")
    if args["train_microbatch_size"] == 1:
        # Check that the pre-microbatch configuration still opens on resume.
        path = interrupted / "training_config.json"
        settings = json.loads(path.read_text())
        settings.pop("train_microbatch_size")
        path.write_text(json.dumps(settings))
    t1_train.train(SimpleNamespace(**args, output=interrupted, variant=variant, resume=True))
    artifacts = ["adapter/adapter_model.safetensors"]
    if variant == "spatial":
        artifacts.append("spatial.safetensors")
    for name in artifacts:
        restored = load_file(interrupted / name)
        for key, expected in load_file(reference / name).items():
            torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)
    with (interrupted / "loss.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    total = min(10, args.get("max_updates", 10) * args["accumulation"])
    count = math.ceil(total / args["accumulation"])
    updates = list(range(1, count + 1))
    assert [int(row["update"]) for row in rows] == updates
    assert [int(row["examples"]) for row in rows] == [min(total, i * args["accumulation"]) for i in updates]
    assert [path.name for path in t1_train.checkpoints(interrupted / "checkpoints")] == [
        f"step-{step:08d}" for step in updates[-2:]]


def verify_initialization(root, args, variant):
    source = root / f"{variant}-reference"
    expected_adapter = load_file(source / "adapter/adapter_model.safetensors")
    expected_memory = load_file(source / "spatial.safetensors") if variant == "spatial" else None
    original = t1_train.save_checkpoint
    tested = []

    def inspect_initial_weights(path, policy, optimizer, scheduler, progress):
        if progress["updates"] == 0:
            adapter = get_peft_model_state_dict(policy.model)
            assert adapter.keys() == expected_adapter.keys()
            for key, value in adapter.items():
                torch.testing.assert_close(value.detach().cpu(), expected_adapter[key], rtol=0, atol=0)
            if expected_memory is not None:
                for key, value in policy.memory.state_dict().items():
                    torch.testing.assert_close(value.detach().cpu(), expected_memory[key], rtol=0, atol=0)
            assert any("lora_" in name and value.requires_grad for name, value in policy.model.named_parameters())
            tested.append(True)
        original(path, policy, optimizer, scheduler, progress)

    settings = dict(args, initialize_from=source)
    with patch.object(t1_train, "save_checkpoint", inspect_initial_weights):
        t1_train.train(SimpleNamespace(**settings, output=root / f"{variant}-initialized",
                                       variant=variant, resume=False))
    assert tested == [True]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory() as folder:
        root = Path(folder)
        tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
        dataset, rows = make_dataset(root, tokenizer.null_token())
        verify_memory_math()
        verify_context(tiny_policy(args.checkpoint, "long"), rows)
        verify_context(tiny_policy(args.checkpoint, "spatial"), rows)
        for variant in ("no_memory", "spatial"):
            verify_microbatch(tiny_policy(args.checkpoint, variant), rows)
        print("Official prompt, cached visual, batched logits and batched greedy decode checks passed")
        print("Spatial geometry and both training microbatch gradients passed")
        common = dict(checkpoint=args.checkpoint, dataset=dataset, epochs=1, accumulation=2,
                      learning_rate=1e-4, budget_seconds=0, resolution=None, seed=88,
                      quantized=False, validate_every=0, cache_vision=True, archive_cache=None,
                      validation_batch_size=1, checkpoint_every=1,
                      initialize_from=None, history_dropout=0.5)
        with patch.object(t1_train, "T1Policy", tiny_policy):
            for variant in ("no_memory", "spatial"):
                for batch_size in (1, 2):
                    run = root / f"microbatch-{batch_size}"
                    run.mkdir(exist_ok=True)
                    batch_settings = dict(common, train_microbatch_size=batch_size,
                                          accumulation=2 if batch_size == 1 else 3)
                    recover(run, batch_settings, variant)
                    if batch_size == 1:
                        verify_initialization(run, batch_settings, variant)
            run = root / "balanced"
            run.mkdir()
            settings = dict(common, train_microbatch_size=2, balance_sources=True, max_updates=4)
            recover(run, settings, "spatial")
            run = root / "steering-weighted"
            run.mkdir()
            turn = "".join(f"<|reserved_special_token_{n}|>" for n in (178, 219, 250, 179))
            steering_dataset, _ = make_dataset(run, tokenizer.null_token(), turn_label=turn)
            weight = t1_train.visible_control_weights(tiny_policy(args.checkpoint, "spatial"), 3.0)
            assert weight(VISIBLE_INSTRUCTION, turn) == 3.0
            assert weight(INSTRUCTION, turn) == weight(VISIBLE_INSTRUCTION, tokenizer.null_token()) == 1.0
            recover(run, dict(settings, dataset=steering_dataset, visible_turn_weight=3.0), "spatial")
        failing = iter([TimeoutError("injected timeout")] * 5 + [42])

        def storage_probe():
            item = next(failing)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("t1_storage.sleep") as pauses:
            assert retry_read(storage_probe) == 42
            assert [call.args[0] for call in pauses.call_args_list] == [1, 2, 4, 8, 16]
        print("Exact initialization, resumed weights, log rollback, checkpoint retention and backoffs passed")


if __name__ == "__main__":
    main()

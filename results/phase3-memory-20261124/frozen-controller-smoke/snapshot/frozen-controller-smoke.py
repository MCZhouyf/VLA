"""Verify that a warm-start spatial run updates only memory parameters."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from jarvisvla.inference.action_mapping import OneActionTokenizer
from safetensors.torch import load_file

import t1_train
import t1_trained_eval
from test_t1_resume import make_dataset, tiny_policy


def check(checkpoint):
    with TemporaryDirectory() as folder, patch.object(t1_train, "T1Policy", tiny_policy):
        root = Path(folder)
        tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
        dataset, _ = make_dataset(root, tokenizer.null_token())
        common = dict(checkpoint=checkpoint, dataset=dataset, archive_cache=None, cache_vision=True,
                      budget_seconds=0, validate_every=0, checkpoint_every=1, resume=False,
                      epochs=1, accumulation=2, train_microbatch_size=2, resolution=None,
                      seed=42, quantized=False, learning_rate=1e-3, history_dropout=1.0,
                      max_updates=3, validation_batch_size=1)
        for variant in ("no_memory", "spatial"):
            t1_train.train(SimpleNamespace(**common, variant=variant, output=root / variant,
                                           initialize_from=None))
        output = root / "frozen"
        t1_train.train(SimpleNamespace(**common, variant="spatial", output=output,
                                       initialize_from=root / "spatial", frozen_controller_run=root / "no_memory",
                                       position_value_scale=0.5))
        settings = json.loads((output / "training_config.json").read_text())
        adapter = Path("adapter/adapter_model.safetensors")
        actual = load_file(output / adapter)
        expected = load_file(root / "no_memory" / adapter)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        memory = load_file(output / "spatial.safetensors")
        before = load_file(root / "spatial/spatial.safetensors")
        changed = sum(not torch.equal(memory[key], before[key]) for key in memory)
        assert changed and settings["trainable_parameters"] == sum(value.numel() for value in memory.values())
        assert settings["position_value_scale"] == 0.5
        with patch.object(t1_trained_eval, "T1Policy", tiny_policy):
            restored = t1_trained_eval.Controller(checkpoint, output, device="cpu").policy.memory
        assert restored.position_value_scale == 0.5
        for name, restored_value in restored.state_dict().items():
            torch.testing.assert_close(restored_value, memory[name], atol=0, rtol=0)
        return dict(lora_tensors_unchanged=len(actual), memory_tensors_changed=changed,
                    trainable_parameters=settings["trainable_parameters"],
                    spatial_updates=json.loads((output / "summary.json").read_text())["updates"],
                    evaluation_reload="exact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.checkpoint), indent=2) + "\n")

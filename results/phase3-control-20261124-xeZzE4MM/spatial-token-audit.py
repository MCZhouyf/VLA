"""Identify disagreed token positions on four short validation inputs without using a GPU."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from t1_model import T1Policy
from t1_train import case_examples, mask_history


root = Path(sys.argv[1])
training = root/"repaired/train-spatial"
config = json.loads((training/"training_config.json").read_text())
index = (root/"mix/dataset.jsonl").read_bytes()
assert hashlib.sha256(index).hexdigest() == config["dataset_sha256"]
assert config["evaluation_history"] == "placeholder"
cache = Path((root/"mix/archive_cache").read_text().strip())
records = []
frequency = Counter()
for line in index.splitlines():
    row = json.loads(line)
    if row["split"] != "validation" or not row["selected_labels"]:
        continue
    with ZipFile(cache/Path(row["path"]).name) as archive:
        labels = json.loads(archive.read("episode.json"))["action"]
    frequency.update(labels[row["outbound_frames"]:row["selected_frames"]])
    records.append((row, labels))
label, count = frequency.most_common(1)[0]
assert sum(frequency.values()) == 2401
cases = []
for row, labels in records:
    if row["group"] != "visible":
        continue
    positions = [i for i in range(row["outbound_frames"], row["selected_frames"]) if labels[i] == label]
    if positions:
        cases.append((row, positions[0]))
cases.sort(key=lambda pair: (pair[1], pair[0]["scene_seed"]))
cases = cases[:4]
assert len(cases) == 4
states = {step: training/"checkpoints"/f"step-{step:08d}" for step in (1000, 2000)}
for folder in states.values():
    assert (folder/"spatial.safetensors").is_file() and (folder/"adapter/adapter_model.safetensors").is_file()
torch.set_num_threads(16)
model = T1Policy(Path(config["checkpoint"]), "spatial", adapter=states[1000]/"adapter",
                 device="cpu", dtype=torch.float32)
model.eval()
tokenizer = model.processor.tokenizer
examples = []
for row, index_in_episode in cases:
    source = dict(row, selected_frames=index_in_episode+1,
                  selected_labels=index_in_episode+1-row["outbound_frames"])
    indices = list(range(row["outbound_frames"], index_in_episode+1))
    random.Random(config["seed"]+row["scene_seed"]).shuffle(indices)
    example = next(case_examples(model, source, config["seed"], config.get("resolution"),
                                 cache_vision=True, archive_cache=cache, skip=indices.index(index_in_episode)))
    assert example[3] == label
    examples.append(mask_history(example, model.actions.null_token()))

result = dict(device="cpu", dtype="float32", selection="four short visible validation episodes with modal target",
              modal_target_validation_frequency=count, modal_target=label,
              validation_actions=sum(frequency.values()), checkpoints={})
with torch.inference_mode():
    for step, folder in states.items():
        if step != 1000:
            state = load_file(folder/"adapter/adapter_model.safetensors")
            matched = set_peft_model_state_dict(model.model, state)
            assert not matched.unexpected_keys
        model.memory.load_state_dict(load_file(folder/"spatial.safetensors"))
        outcomes = []
        for (row, i), (logits, targets) in zip(cases, model.batched_action_logits(examples), strict=True):
            expect = targets.tolist()
            prediction = logits.argmax(dim=-1).tolist()
            mismatches = [{"position": n, "expected_id": a, "predicted_id": b,
                           "expected": tokenizer.decode([a]), "predicted": tokenizer.decode([b])}
                          for n, (a, b) in enumerate(zip(expect,prediction,strict=True)) if a != b]
            outcomes.append(dict(seed=row["scene_seed"], action_index=i, mismatches=mismatches))
        result["checkpoints"][step] = outcomes
        print(step, json.dumps(outcomes, ensure_ascii=False), flush=True)
path = root/"spatial-token-audit.json"
with path.open("x") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
    f.write("\n")

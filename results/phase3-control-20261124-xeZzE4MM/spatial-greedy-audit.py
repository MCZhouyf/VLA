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
                 device="cuda", dtype=torch.bfloat16)
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


from minestudio.simulator.entry import CameraConfig, MinecraftSim
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.actions import ActionTransformer
from phase1_t1 import CAMERA

camera = CameraConfig(**CAMERA)
sim = MinecraftSim.__new__(MinecraftSim)
sim.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera.n_camera_bins)
sim.action_transformer = ActionTransformer(**camera.action_transformer_kwargs)

def environmental(agent_action):
    decoded = sim.agent_action_to_env_action(agent_action)
    return dict(forward=int(decoded["forward"]), camera=decoded["camera"].tolist())

requests = [(a,b,c,e,f) for a,b,c,_,e,f in examples]
target_ids = model.action_ids(label)
teacher = environmental(model.actions.decode(target_ids)[0])
result = dict(dtype="bf16", physical_gpu=7, teacher_tokens=target_ids, teacher_environment=teacher,
              selection="the same four visible modal-target validation inputs used in the FP32 audit",
              checkpoints={})
with torch.inference_mode():
    for step, folder in states.items():
        if step != 1000:
            state = load_file(folder/"adapter/adapter_model.safetensors")
            matched = set_peft_model_state_dict(model.model,state)
            assert not matched.unexpected_keys
        model.memory.load_state_dict(load_file(folder/"spatial.safetensors"))
        group = model.predict_batch(requests, return_ids=True)
        single = [model.predict(*request, return_ids=True) for request in requests]
        logits = list(model.batched_action_logits(examples))
        assert len(group) == len(single) == len(logits) == len(cases)
        outcomes = []
        for (row,i),one,many,(scores,targets) in zip(cases,single,group,logits,strict=True):
            assert targets.tolist() == target_ids
            outcomes.append(dict(seed=row["scene_seed"], action_index=i,
                                 teacher_forced_tokens=scores.argmax(dim=-1).tolist(),
                                 greedy_single_tokens=one[3], greedy_batch_tokens=many[3],
                                 single_environment=environmental(one[0]), batch_environment=environmental(many[0]),
                                 complete_single=bool(one[2]), complete_batch=bool(many[2])))
        result["checkpoints"][step]=outcomes
        print(step,json.dumps(outcomes),flush=True)
with (root/"spatial-greedy-audit.json").open("x") as output:
    json.dump(result,output,indent=2,ensure_ascii=False)
    output.write("\n")

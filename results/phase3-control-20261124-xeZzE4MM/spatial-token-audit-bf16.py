"""Read-only analysis of the archived spatial checkpoint on the complete validation split."""
import hashlib
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import torch
from safetensors.torch import load_file

from t1_model import T1Policy
from t1_train import case_examples, mask_history

run = Path(sys.argv[1])
training = run/"repaired/train-spatial"
state = training/"checkpoints/step-00002000"
cfg = json.loads((training/"training_config.json").read_text())
cache = Path((run/"mix/archive_cache").read_text().strip())
frozen = (run/"mix/dataset.jsonl").read_bytes()
assert hashlib.sha256(frozen).hexdigest() == cfg["dataset_sha256"]
assert cfg["evaluation_history"] == "placeholder"
validation = [r for r in (json.loads(line) for line in frozen.splitlines()) if r["split"] == "validation"]
frames = sum(r["selected_frames"] for r in validation)
assert frames == 6500 and sum(r["selected_labels"] for r in validation) == 2401
model = T1Policy(Path(cfg["checkpoint"]), "spatial", adapter=state/"adapter", device="cuda", dtype=torch.bfloat16)
model.memory.load_state_dict(load_file(state/"spatial.safetensors"))
model.eval()

errors = Counter()
wrong_per_action = Counter()
substitutions = Counter()
total_tokens = correct_tokens = exact_actions = changed_actions = changed_exact = 0
with torch.inference_mode():
    examples = (example for row in validation
                for example in case_examples(model, row, cfg["seed"], cfg.get("resolution"),
                                             cache_vision=True, archive_cache=cache))
    while group := list(islice(examples, cfg["validation_batch_size"])):
        masked = [mask_history(example, model.actions.null_token()) for example in group]
        preds = (model.batched_action_logits(masked) if len(masked) > 1
                 else [model.action_logits(*masked[0])])
        for example, (logits, targets) in zip(group, preds, strict=True):
            pred, target = logits.argmax(dim=-1).tolist(), targets.tolist()
            mismatches = [(pos,a,b) for pos,(a,b) in enumerate(zip(target,pred,strict=True)) if a != b]
            total_tokens += len(target)
            correct_tokens += len(target)-len(mismatches)
            exact_actions += not mismatches
            changed = example[3] != example[2][-1]
            changed_actions += changed
            changed_exact += changed and not mismatches
            wrong_per_action[len(mismatches)] += 1
            for pos,a,b in mismatches:
                errors[pos] += 1
                substitutions[(pos,a,b)] += 1
assert sum(wrong_per_action.values()) == 2401 and total_tokens == 11751 and changed_actions == 143
tokenizer = model.processor.tokenizer
result = dict(checkpoint=2000, physical_gpu=7, dtype="bf16", frames=frames,
              actions=sum(wrong_per_action.values()), tokens=total_tokens,
              correct_tokens=correct_tokens, exact_actions=exact_actions,
              changed_actions=changed_actions, changed_exact=changed_exact,
              wrong_tokens_per_action=dict(wrong_per_action), errors_by_position=dict(errors),
              top_substitutions=[dict(position=pos, expected_id=a, predicted_id=b,
                                      expected=tokenizer.decode([a]), predicted=tokenizer.decode([b]), count=n)
                                 for (pos,a,b),n in substitutions.most_common(20)])
with (run/"spatial-token-audit-bf16.json").open("x") as output:
    json.dump(result,output,indent=2,ensure_ascii=False)
    output.write("\n")
print(json.dumps(result,ensure_ascii=False),flush=True)

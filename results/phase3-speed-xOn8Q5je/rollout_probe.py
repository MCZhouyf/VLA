import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvisvla.inference.action_mapping import OneActionTokenizer
from t1_occlusion import set_candidate_render_distance
from t1_trained_eval import drive_rollouts, screened_rollout

class ProbeController:
    def start(self, outbound):
        assert outbound

output = Path.cwd()
set_candidate_render_distance(output)
config = json.loads(Path(sys.argv[1]).read_text())["fixtures"]
tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
noop = tokenizer.decode([])[0]
seen_widths = []

def control(requests):
    seen_widths.append(len(requests))
    return [(noop, True) for _ in requests]

cases = []
for seed in (2026112105, 2026112106):
    fixture = dict(config[f"30/{seed}"], return_limit=3)
    cases.append(screened_rollout(SimpleNamespace(output=output), seed, 30, fixture,
                                  ProbeController()))
results = {}
with patch("t1_trained_eval.choose_actions", control):
    drive_rollouts(cases, lambda k, v: results.__setitem__(k, v))
assert seen_widths == [2, 2, 2], seen_widths
assert all(not result[0][0] and result[0][1] == 3 and result[0][-1] == "budget"
           for result in results.values()), results
print("Two screened environments shared three rollout batches with isolated traces")

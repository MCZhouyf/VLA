import json
import sys
from pathlib import Path
from time import monotonic
from zipfile import ZipFile

import torch
from t1_model import T1Policy
from t1_train import load_images

work, run = map(Path, sys.argv[1:])
with (work/"data/t1-screened-hzw7nYp7/index-full/dataset.jsonl").open() as stream:
    rows = [json.loads(line) for line in stream]
cache = Path((run/"disk_cache_path").read_text().strip())
visible = [r for r in rows if r["split"]=="validation" and str(r["group"])=="visible"
           and r["outbound_frames"]==2 and r["selected_labels"]>0]
visible.sort(key=lambda r:r["scene_seed"])
if not visible:
    raise RuntimeError("No visible validation episodes")
policy = T1Policy(work/"models/JarvisVLA-Qwen2-VL-7B", "no_memory",
                  adapter=run/"train-no_memory"/"adapter", device="cpu", dtype=torch.float32)
policy.eval()
def serialize(control):
    return {k: v.tolist() if hasattr(v, "tolist") else v for k,v in control.items()}
for row in visible[:2]:
    elapsed = monotonic()
    with ZipFile(cache/Path(row["path"]).name) as archive:
        metadata=json.loads(archive.read("episode.json"))
        frames=load_images(archive,range(3), policy.images, None)
    expected = metadata["action"][2]
    encoder=policy.processor.tokenizer
    encoded=policy.encode_frames(frames)
    previous=metadata["action"][:2]
    with torch.inference_mode():
        logits, targets=policy.action_logits(metadata["instruction"], encoded, previous, expected)
        own_tokens=logits.argmax(-1).tolist()
        control, decoded, complete=policy.predict(metadata["instruction"], encoded, previous)
    print(json.dumps(dict(seed=row["scene_seed"], device="CPU", precision="FP32",
          ground_truth=expected, expected_control=serialize(policy.actions.decode(policy.action_ids(expected))[0]),
          teacher_forced_tokens=encoder.decode(own_tokens), teacher_forced_exact=own_tokens==targets.tolist(),
          greedy=decoded, greedy_control=serialize(control), complete=complete,
          seconds=monotonic()-elapsed)), flush=True)

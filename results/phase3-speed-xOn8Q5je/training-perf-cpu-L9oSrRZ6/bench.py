import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from t1_spatial import CELL_BLOCKS, SIDE, HEADINGS, retained_cells, write_weights

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("original_spatial", root/"code/minemem/t1_spatial.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
cache = Path(sys.argv[2])
dataset = [json.loads(line) for line in (root/"data/t1-screened-hzw7nYp7/index-full/dataset.jsonl").read_text().splitlines()]
rows = [row for row in dataset if row["split"] == "train" and row["selected_labels"]]
n_frames = sum(r["selected_frames"] for r in rows)
early = sum(max(0, r["outbound_frames"] - 2) // 3 * 3 for r in rows)
data = dict(episodes=len(rows), training_labels=sum(r["selected_labels"] for r in rows),
            no_memory_vision_frames=sum(r["selected_frames"] - max(0, r["outbound_frames"] - 2) for r in rows),
            spatial_vision_frames=n_frames, early_frames_pooled_before_copy=early)
variant_group = {}
for r in rows:
    variant_group.setdefault(r["group"], []).append(r)
rand = random.Random(88)
prefixes = []
for group in variant_group.values():
    for r in rand.sample(group, min(10, len(group))):
        with ZipFile(cache/Path(r["path"]).name) as f:
            pose=np.asarray(json.loads(f.read("episode.json"))["pose"], dtype=np.float32)
        for step in np.linspace(r["outbound_frames"], r["selected_frames"]-1, 6, dtype=int):
            prefixes.append(pose[:step+1])
inputs = []
rng=np.random.default_rng(93)
for poses in prefixes:
    before=old.retained_cells(poses)
    after=retained_cells(poses)
    for a,b in zip(before,after,strict=True):
        np.testing.assert_array_equal(a,b)
    cells, origin, keep = after
    if keep.any():
        loc=cells[keep]-origin
        bearings=np.floor((poses[:-1,2][keep]%360)*HEADINGS/360).astype(np.int64)
        keys=(loc[:,0]*SIDE+loc[:,1])*HEADINGS+bearings
        slots, inverse = torch.unique(torch.from_numpy(keys),return_inverse=True)
        gate=torch.from_numpy(rng.uniform(.01,.99,len(keys)).astype(np.float32))
        inputs.append((gate,inverse,len(slots)))

def dense(gate,inverse,_):
    pair=(inverse[:,None]==inverse[None,:]).triu(1).float()
    return gate*torch.exp(pair@torch.log1p(-gate))

def run(time_function,samples):
    stats=[]
    for _ in range(4):
        began=perf_counter()
        for args in samples:
            time_function(*args)
        stats.append(perf_counter()-began)
    return statistics.median(stats)

with torch.inference_mode():
    for args in inputs:
        torch.testing.assert_close(dense(*args),write_weights(*args), rtol=5e-5,atol=1e-6)
    before=run(old.retained_cells,[(p,) for p in prefixes])
    after=run(retained_cells,[(p,) for p in prefixes])
    weights_before=run(dense,inputs)
    weights_after=run(write_weights,inputs)
print(json.dumps(dict(**data,cpu_prefixes=len(prefixes),
    prefix_grid_before_ms=1000*before,prefix_grid_after_ms=1000*after,
    weight_cases=len(inputs),weight_dense_ms=1000*weights_before,weight_segmented_ms=1000*weights_after,
    ratio_grid=before/after,ratio_weights=weights_before/weights_after), indent=2))

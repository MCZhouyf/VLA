import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import t1_train
from test_t1_resume import tiny_policy


def main():
    import sys
    root, weights = map(Path, sys.argv[1:])
    with TemporaryDirectory(dir=root) as tmp, patch.object(t1_train, "T1Policy", tiny_policy):
        scratch = Path(tmp)
        rows = []
        for role in ("train","dart"):
            folder = root / f"smoke-{role}/worker-000"
            row = json.loads((folder/"episodes.jsonl").read_text().splitlines()[0])
            rows.append(dict(row, path=str(folder/row["path"]), selected_frames=row["outbound_frames"]+4,
                             selected_labels=4))
        dataset = scratch / "dataset.jsonl"
        dataset.write_text("".join(json.dumps(row)+"\n" for row in rows))
        oracle = tiny_policy(weights, "no_memory")
        for row in rows:
            examples=list(t1_train.case_examples(oracle, row, 123, [112,84], cache_vision=True))
            assert len(examples)==4
            assert all(sample[0].startswith("Follow the waypoint.\nwaypoint: ") for sample in examples)
        options = dict(checkpoint=weights, dataset=dataset, archive_cache=None, cache_vision=True,
                       budget_seconds=0, validate_every=0, checkpoint_every=1, resume=False,
                       epochs=1, accumulation=4, train_microbatch_size=2, resolution=[112,84],
                       seed=42, quantized=False, learning_rate=1e-3, history_dropout=.5,
                       max_updates=2, validation_batch_size=1, variant="no_memory",
                       output=scratch/"output", initialize_from=None)
        t1_train.train(SimpleNamespace(**options))
        result=json.loads((scratch/"output/summary.json").read_text())
        assert result["updates"]==2 and result["supervised_examples"]==8
        (root/"controller-training-smoke.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()

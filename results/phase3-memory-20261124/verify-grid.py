import csv,gzip,json,sys
from pathlib import Path
import numpy as np
from t1_spatial import retained_cells
from t1_memory_audit import matching_traces,retained_indices

work=Path.cwd()
root=Path(sys.argv[1])
fixture_path=work/"results/phase3-speed-xOn8Q5je/screened_fixtures.json"
fixtures=json.loads(fixture_path.read_text())["fixtures"]
review=work/"results/t1-review-A8vp1K1p"
trace_roots=list(review.glob("screened-acceptance*"))+[review/"alternate-final-seed",
    work/"results/phase3-control-20261124-xeZzE4MM/diagnostic-d30-no_memory"]
matched=matching_traces(fixtures.values(),[p for p in trace_roots if p.is_dir()])
kept_checked=0
for name,(trace,fixture) in matched.items():
    with gzip.open(trace,"rt") as f:
        poses=[tuple(float(row[key]) for key in ("x","z","yaw")) for row in csv.DictReader(f)]
    poses[-1]=tuple(float(fixture["return_pose"][key]) for key in ("x","z","yaw"))
    actual_cells,_,mask=retained_cells(np.asarray(poses,dtype=np.float32))
    cells,kept=retained_indices(poses,32)
    assert kept==set(np.flatnonzero(mask)),name
    assert np.array_equal(cells[:-1],actual_cells),name
    kept_checked+=len(kept)
result=dict(frozen_scenarios=len(matched),matched_retained_indices=len(matched),retained_frames_checked=kept_checked)
(root/"verified-grid.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))

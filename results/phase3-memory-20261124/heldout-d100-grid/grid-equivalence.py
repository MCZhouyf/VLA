import importlib.util
from pathlib import Path
import numpy as np
import torch
import t1_spatial

old=Path.cwd()/"code/minemem/t1_spatial.py"
spec=importlib.util.spec_from_file_location("unchanged_spatial",old)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(4)
torch.manual_seed(73)
default=module.SpatialMemory(64,64,width=32).eval()
new=t1_spatial.SpatialMemory(64,64,width=32).eval()
wide=t1_spatial.SpatialMemory(64,64,width=32,grid_side=64).eval()
assert new.grid_side==32 and wide.grid_side==64
new.load_state_dict(default.state_dict())
wide.load_state_dict(default.state_dict())
short=[(float(x),0.,90.) for x in range(20)]
long=[(float(x),0.,90.) for x in range(90)]
with torch.inference_mode():
    for path in (short,long):
        frames=torch.randn(len(path)-1,64)
        torch.testing.assert_close(new(frames,path),default(frames,path),rtol=0,atol=0)
        if path is short:
            torch.testing.assert_close(wide(frames,path),default(frames,path),rtol=0,atol=0)
coords=np.asarray(long,dtype=np.float32)
assert not t1_spatial.retained_cells(coords,32)[2][0]
assert t1_spatial.retained_cells(coords,64)[2][0]
print("Default output and shared local prefix exactly match; wider grid retains the far start")

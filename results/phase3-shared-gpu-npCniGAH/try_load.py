import json
import sys
import torch

from t1_model import T1Policy

checkpoint, mode = sys.argv[1:]
assert torch.cuda.device_count() == 1
free, total = torch.cuda.mem_get_info()
cap = min(768 * 2**20, free - 512 * 2**20)
assert cap >= 128 * 2**20, f"No shareable memory after retaining headroom ({free/2**20:.2f} MiB)"
torch.cuda.set_per_process_memory_fraction(cap/total, 0)
print(json.dumps(dict(mode=mode, free_mib=round(free/2**20,2),
                      allocator_cap_mib=round(cap/2**20,2))), flush=True)
T1Policy(checkpoint, "no_memory", quantized=mode == "nf4", training=True)
print("7B LoRA backbone loaded", flush=True)

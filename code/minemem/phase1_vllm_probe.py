"""Check concurrent inference on the independent per-GPU vLLM instances."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import perf_counter

import numpy as np
from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT


def probe(endpoint, checkpoint, requests, barrier):
    agent = VLLM_AGENT(checkpoint_path=str(checkpoint), base_url=endpoint, history_num=2,
                      temperature=0.9, action_chunk_len=1, instruction_type="normal")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    barrier.wait(timeout=120)
    start = perf_counter()
    for _ in range(requests):
        result = agent.forward([frame], ["mine_block:stone"])
        assert "buttons" in result and "camera" in result
    seconds = perf_counter() - start
    return endpoint, requests, seconds, requests / seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("endpoints", nargs="+")
    parser.add_argument("--requests", type=int, default=10)
    args = parser.parse_args()
    if args.requests < 1 or len(set(args.endpoints)) != len(args.endpoints):
        parser.error("Provide distinct endpoints and a positive request count")
    barrier = Barrier(len(args.endpoints))
    with ThreadPoolExecutor(max_workers=len(args.endpoints)) as pool:
        futures = [pool.submit(probe, address, args.checkpoint, args.requests, barrier)
                   for address in args.endpoints]
        results = [future.result() for future in futures]
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint", "completed_requests", "seconds", "requests_per_second"))
        writer.writerows(results)


if __name__ == "__main__":
    main()

"""Poll allocated GPUs and run one MineMem Phase 0 MCU rollout."""

import argparse
import csv
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


POLL_SECONDS = 60
MAX_IDLE_MIB = 256
TEMPORARY_FAILURE = 75
MCU_CHECKPOINT_SIZES = {
    "model-00001-of-00004.safetensors": 4966659480,
    "model-00002-of-00004.safetensors": 4991495688,
    "model-00003-of-00004.safetensors": 4932750920,
    "model-00004-of-00004.safetensors": 1691924368,
}
RECORD_FIELDS = ("time", "gpu", "status", "used_mib", "util_percent",
                 "compute_processes", "phase0_status")
INSTALL_SPECS = (
    "numpy==1.26.4", "torch==2.5.1", "transformers==4.49.0",
    "accelerate==1.2.1", "huggingface-hub==0.28.1",
)


def query_gpu(gpu):
    def query(field):
        return subprocess.check_output(
            ["nvidia-smi", "-i", gpu, field, "--format=csv,noheader,nounits"],
            text=True, timeout=8, stderr=subprocess.PIPE,
        ).strip()

    try:
        name, memory, util = map(str.strip, query(
            "--query-gpu=name,memory.used,utilization.gpu").split(","))
        if "A100" not in name or "40GB" not in name:
            return "unexpected_device", "", "", ""
        memory, util = int(memory), int(util)
        count = len(query("--query-compute-apps=pid").splitlines())
    except (OSError, subprocess.SubprocessError, ValueError):
        return "query_error", "", "", ""
    idle = count == 0 and util == 0 and memory <= MAX_IDLE_MIB
    return ("idle" if idle else "busy"), memory, util, count


def random_passed(work):
    for path in (work / "results").glob("phase0-random-*/summary.csv"):
        with path.open(newline="") as stream:
            if any(row["success"] == "True" and row["steps"] == "100"
                   for row in csv.DictReader(stream)):
                return True
    return False


def checkpoint_present(work):
    model = work / "models/JarvisVLA-Qwen2-VL-7B"
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        if not (model / name).is_file():
            return False
    try:
        return all((model / name).stat().st_size == size
                   for name, size in MCU_CHECKPOINT_SIZES.items())
    except FileNotFoundError:
        return False


class Preparation:
    def __init__(self, work):
        self.work = work
        self.entry = work / "code/minemem/phase0.sh"
        self.environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        self.install = None
        self.random_run = None

    def start(self, command, name):
        path = self.work / "results" / f"phase0-watch-{name}.log"
        with path.open("a") as log:
            return subprocess.Popen(command, cwd=self.work, env=self.environment,
                                    stdout=log, stderr=subprocess.STDOUT)

    def status(self):
        work = self.work
        mine = work / "code/MineStudio"
        if not all((mine / name).is_file() for name in
                   ("pyproject.toml", "minestudio/simulator/entry.py")):
            return "waiting_mine_studio_source"

        if self.install is not None and self.install.poll() is None:
            return "installing_dependencies"
        probe = (
            "from importlib.util import find_spec\n"
            "names = ('minestudio', 'cv2', 'torch', 'transformers', 'accelerate', 'yaml')\n"
            "raise SystemExit(not all(find_spec(name) for name in names))"
        )
        installed = subprocess.run([sys.executable, "-c", probe], env=self.environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not installed:
            if self.install is not None:
                return "dependency_install_failed"
            self.install = self.start([sys.executable, "-m", "pip", "install",
                                       "--progress-bar", "off", *INSTALL_SPECS, "-e", str(mine)],
                                      "dependencies")
            return "installing_dependencies"

        if not random_passed(work):
            if self.random_run is None:
                self.random_run = self.start(["bash", str(self.entry), "random"], "random")
            elif self.random_run.poll() is not None:
                return "random_smoke_failed"
            return "running_random_smoke"

        if not (work / "code/JarvisVLA/jarvisvla/inference/action_mapping.py").is_file():
            return "waiting_jarvis_source"
        if not checkpoint_present(work):
            return "waiting_jarvis_checkpoint"
        return "ready"


def monitor(work, gpus):
    results = work / "results"
    results.mkdir(parents=True, exist_ok=True)
    temp = Path(os.environ["TMPDIR"])
    temp.mkdir(parents=True, exist_ok=True)
    with (temp / "phase0-gpu-watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A Phase 0 GPU watcher is already running.")
            return 0
        path = results / "phase0_gpu_watch.csv"
        preparation = Preparation(work)
        with path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            if path.stat().st_size == 0:
                writer.writerow(RECORD_FIELDS)
            previous = None
            while True:
                status = preparation.status()
                observations = {gpu: query_gpu(gpu) for gpu in gpus}
                states = [item[0] for item in observations.values()]
                now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
                for gpu, item in observations.items():
                    writer.writerow((now, gpu, *item, status))
                stream.flush()
                if (status, states) != previous:
                    print(now, status, *(f"GPU {gpu}: {state}" for gpu, state in zip(gpus, states)))
                    previous = status, states
                if status == "ready" and "idle" in states:
                    selected, spare = sorted(gpus, key=lambda gpu: observations[gpu][0] != "idle")
                    environment = dict(os.environ, GPU_IDS=f"{selected},{spare}",
                                       CUDA_VISIBLE_DEVICES=f"{selected},{spare}",
                                       MINEMEM_REQUIRE_IDLE="1")
                    print(f"Starting the MCU rollout on GPU {selected}.")
                    result = subprocess.run(["bash", str(preparation.entry), "mcu"],
                                            cwd=work, env=environment)
                    if result.returncode != TEMPORARY_FAILURE:
                        print(f"MCU rollout exited with status {result.returncode}.")
                        return result.returncode
                    print(f"GPU {selected} became busy or could not be verified. Polling again.")
                time.sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gpus = os.environ["GPU_IDS"].split(",")
    parser.add_argument("--check-idle", choices=gpus)
    args = parser.parse_args()
    if args.check_idle is not None:
        return 0 if query_gpu(args.check_idle)[0] == "idle" else TEMPORARY_FAILURE
    return monitor(Path(os.environ["WORK"]), gpus)


if __name__ == "__main__":
    raise SystemExit(main())

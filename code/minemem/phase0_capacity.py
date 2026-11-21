"""Measure concurrent CPU MineStudio environment throughput."""

import argparse
import csv
import json
import os
import select
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from time import perf_counter, sleep


def write_rows(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stop_worker(signum, frame):
    raise SystemExit(128 + signum)


def worker(batch, directory, seed, count):
    signal.signal(signal.SIGTERM, stop_worker)
    import cv2
    from minestudio.simulator import MinecraftSim

    cv2.setNumThreads(1)
    init_time = perf_counter()
    sim = MinecraftSim(action_type="env", seed=seed, preferred_spawn_biome="plains")
    try:
        sim.reset()
        space = sim.action_space
        space.seed(seed)
        setup_seconds = perf_counter() - init_time
        (directory / "ready").touch()
        while not (batch / "start").exists():
            sleep(0.02)
        started = perf_counter()
        steps = 0
        for _ in range(count):
            _, _, terminated, truncated, _ = sim.step(space.sample())
            steps += 1
            if terminated or truncated:
                break
        ended = perf_counter()
        row = {"worker": directory.name, "seed": seed, "steps": steps, "start": started,
               "end": ended, "startup_seconds": setup_seconds}
        (directory / "step_metrics.json").write_text(json.dumps(row))
    finally:
        sim.close()


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def reap_cpu_children(batch):
    """Clean up MineStudio JVMs that started their own sessions."""
    batch = batch.resolve()
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        system_python = Path("/usr/bin/python3")
        if Path(sys.executable).resolve() == system_python.resolve():
            raise RuntimeError("This Linux interpreter lacks pidfd process control")
        subprocess.run([str(system_python), "-B", str(Path(__file__).resolve()),
                        "reap", str(batch)], check=True)
        return

    children = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            if not (entry / "cwd").resolve(strict=True).is_relative_to(batch):
                continue
            pidfd = os.pidfd_open(int(entry.name))
        except OSError:
            continue
        try:
            cwd = (entry / "cwd").resolve(strict=True)
            environment = (entry / "environ").read_bytes().split(b"\0")
            if cwd.is_relative_to(batch) and b"CUDA_VISIBLE_DEVICES=" in environment:
                children.add(pidfd)
            else:
                os.close(pidfd)
        except OSError:
            os.close(pidfd)

    poller = select.poll()
    for pidfd in children:
        poller.register(pidfd, select.POLLIN)
    try:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pidfd in children:
                with suppress(ProcessLookupError):
                    signal.pidfd_send_signal(pidfd, sig)
            deadline = perf_counter() + 10
            while children and perf_counter() < deadline:
                for pidfd, _ in poller.poll(300):
                    poller.unregister(pidfd)
                    children.remove(pidfd)
                    os.close(pidfd)
        if children:
            raise RuntimeError(f"{len(children)} CPU processes remain under {batch}")
    finally:
        for pidfd in children:
            os.close(pidfd)


def run_level(output, level, config):
    batch = output / f"workers-{level:03}"
    batch.mkdir()
    canonical = Path(os.environ["MINESTUDIO_DIR"])
    worker_paths = []
    processes = []
    try:
        for number in range(level):
            directory = batch / f"worker-{number:03}"
            directory.mkdir()
            runtime = directory / "minestudio"
            runtime.mkdir()
            (runtime / "engine").symlink_to(canonical / "engine", target_is_directory=True)
            if (canonical / "runtime").is_dir():
                shutil.copytree(canonical / "runtime", runtime / "runtime")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MINESTUDIO_DIR=str(runtime),
                       LP_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", LIBGL_ALWAYS_SOFTWARE="1",
                       JAVA_TOOL_OPTIONS=os.environ.get("JAVA_TOOL_OPTIONS", "")
                       + " -XX:ActiveProcessorCount=2",
                       MESA_SHADER_CACHE_DIR=str(directory))
            command = ["xvfb-run", "-a", "-e", str(directory / "xvfb.log"),
                       sys.executable, str(Path(__file__).resolve()), "worker", str(batch),
                       str(directory), str(config["seed"] + number), str(config["steps"])]
            with (directory / "worker.log").open("x") as log:
                processes.append(subprocess.Popen(command, cwd=directory, env=env, stdout=log,
                                                  stderr=subprocess.STDOUT, start_new_session=True))
            worker_paths.append(directory)
            sleep(0.15)
        deadline = perf_counter() + config["startup_timeout_seconds"]
        while not all((path / "ready").exists() for path in worker_paths):
            if perf_counter() >= deadline or any(p.poll() is not None for p in processes):
                return {"workers": level, "completed": 0, "steps": 0,
                        "elapsed_seconds": "", "aggregate_fps": "", "status": "startup_failed"}
            sleep(0.1)
        (batch / "start").touch()
        deadline = perf_counter() + config["step_timeout_seconds"]
        while any(process.poll() is None for process in processes) and perf_counter() < deadline:
            sleep(0.1)
    finally:
        stop_processes(processes)
        reap_cpu_children(batch)

    rows = []
    for path in worker_paths:
        evidence = path / "step_metrics.json"
        if evidence.is_file():
            row = json.loads(evidence.read_text())
            row["fps"] = row["steps"] / (row["end"] - row["start"])
            rows.append(row)
    completed = sum(row["steps"] == config["steps"] for row in rows)
    if rows:
        write_rows(batch / "instances.csv", rows)
        seconds = max(row["end"] for row in rows) - min(row["start"] for row in rows)
        steps = sum(row["steps"] for row in rows)
        fps = steps / seconds
    else:
        seconds, steps, fps = "", 0, ""
    clean_exit = all(process.returncode == 0 for process in processes)
    return {"workers": level, "completed": completed, "steps": steps, "elapsed_seconds": seconds,
            "aggregate_fps": fps,
            "status": "complete" if completed == level and clean_exit else "incomplete"}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "reap":
        reap_cpu_children(Path(sys.argv[2]))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        batch, directory, seed, count = sys.argv[2:]
        worker(Path(batch), Path(directory), int(seed), int(count))
        return 0
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("levels", type=int, nargs="*")
    args = parser.parse_args()
    source = Path(__file__).with_name("phase0.yaml")
    settings = yaml.safe_load(source.read_text())
    config = dict(settings["capacity"], seed=settings["seed"])
    if args.levels:
        if any(level < 1 for level in args.levels) or len(set(args.levels)) != len(args.levels):
            parser.error("worker levels must be distinct positive integers")
        config["levels"] = args.levels
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    rows = []
    path = args.output / "throughput.csv"
    for level in config["levels"]:
        row = run_level(args.output, level, config)
        rows.append(row)
        print(row, flush=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        if row["status"] != "complete":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

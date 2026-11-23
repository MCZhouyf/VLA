"""Observe and, when safe, recover the two existing screened Phase 3 runs."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


LANES = {"no_memory": (6, "no-memory"), "spatial": (7, "spatial")}
PARAMETERS = {"--output", "--checkpoint", "--dataset", "--variant", "--epochs",
              "--accumulation", "--seed", "--learning-rate", "--archive-cache",
              "--validate-every", "--validation-batch-size", "--checkpoint-every"}
SWITCHES = {"--cache-vision", "--resume"}
AUTO_RECOVERY = b"attempting one checkpoint recovery"
STORAGE_FAILURE = re.compile(
    rb"TimeoutError|timed out|[Ee]rrno (?:5|11|104|110|116)\b|"
    rb"[Ss]tale file handle|[Cc]onnection reset|[Ii]nput/output error"
)
HARD_FAILURE = re.compile(
    rb"CUDA out of memory|OutOfMemoryError|Nonfinite|Resume settings differ|"
    rb"BadZipFile|FileNotFoundError|PermissionError|No space left|KeyboardInterrupt"
)


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def latest_text(path, size=16384):
    try:
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - size))
            return source.read()
    except FileNotFoundError:
        return b""


def csv_tail(path):
    try:
        with path.open("rb") as source:
            header = source.readline().decode()
            tail_start = max(source.tell(), path.stat().st_size - 16384)
            source.seek(tail_start)
            trail = source.read()
        lines = trail.split(b"\n")[:-1]
        last = {}
        for row in csv.DictReader(io.StringIO(header + b"\n".join(lines[-4:]).decode())):
            if row.get("update", "").isdigit():
                last = row
        return last
    except FileNotFoundError:
        return {}


def exit_status(path):
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def process_args():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(") ", 1)[1]
            if stat.startswith(("Z ", "X ")):
                continue
            argv = [os.fsdecode(arg) for arg in (entry / "cmdline").read_bytes().split(b"\0") if arg]
        except (OSError, IndexError):
            continue
        if argv:
            yield int(entry.name), argv


def same_path(arg, path):
    return arg in (str(path), path.name, f"code/minemem/{path.name}")


def process_roles(processes, work, run, lane):
    roles = {"train": [], "wrapper": [], "recover": [], "evaluation": []}
    output = str(run / f"train-{lane}")
    starter, waiter = (run / f"{prefix}-{LANES[lane][1]}.sh" for prefix in ("start", "after"))
    scripts = (work / "code/minemem")
    for pid, argv in processes:
        if any(same_path(arg, waiter) for arg in argv[:3]) or (
                str(run) in argv and any(same_path(arg, scripts / "t1_screened_eval.sh") for arg in argv[:4])
                and lane in argv):
            roles["evaluation"].append(pid)
        if any(same_path(arg, starter) for arg in argv[:3]):
            roles["wrapper"].append(pid)
        if "--resume-worker" in argv and lane in argv and str(run) in argv and (
                str(scripts / "t1_gpu_supervisor.py") in argv):
            roles["recover"].append(pid)
            continue
        if output not in argv:
            continue
        if len(argv) > 1 and same_path(argv[1], scripts / "t1_recover.sh"):
            roles["wrapper"].append(pid)
            continue
        for index, arg in enumerate(argv[:6]):
            if same_path(arg, scripts / "t1_train.py") and index and "python" in Path(argv[0]).name:
                roles["train"].append((pid, argv[index + 1:]))
                break
    return roles


def query_gpu(gpu):
    def query(fields, expected):
        result = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-{fields}",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=8)
        if result.stderr.strip():
            raise ValueError("GPU query returned a diagnostic")
        rows = list(csv.reader((line for line in result.stdout.splitlines() if line.strip()),
                               skipinitialspace=True))
        if any(len(row) != expected for row in rows):
            raise ValueError("Unexpected GPU query output")
        return rows

    try:
        metrics = query("gpu=index,memory.used,utilization.gpu", 3)
        row, = metrics
        index, used, util = map(int, row)
        if index != gpu:
            raise ValueError("GPU index mismatch")
        pids = [int(pid) for pid, in query("compute-apps=pid", 1)]
        return dict(memory_mib=used, utilization=util, compute_pids=pids,
                    idle=not pids and util == 0 and used <= 256)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(idle=False, error=str(error))


def available_lock(work, gpu):
    path = work / "tmp" / f"phase1-vllm-gpu-{gpu}.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def resolve_args(argv, work, run, lane):
    values, switches, verified = {}, set(), []
    stream = iter(argv)
    for key in stream:
        if key in SWITCHES and key not in switches:
            switches.add(key)
            if key != "--resume":
                verified.append(key)
        elif key in PARAMETERS and key not in values:
            value = next(stream, None)
            if value is None or value.startswith("--"):
                raise ValueError(f"Missing training flag value {key}")
            values[key] = value
            if key != "--checkpoint":
                verified.extend((key, value))
        else:
            raise ValueError(f"Unexpected training flag {key}")
    expected = {
        "--output": str(run / f"train-{lane}"), "--variant": lane,
        "--checkpoint": str(work / "models/JarvisVLA-Qwen2-VL-7B"),
        "--dataset": str(work / "data/t1-screened-hzw7nYp7/index-full/dataset.jsonl"),
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("Live process does not match the frozen screened run")
    if "--archive-cache" not in values:
        raise ValueError("Frozen training must use a verified local archive cache")
    return values, verified


def safetensors_intact(path):
    """Validate safetensors metadata and exact payload length without decoding weights."""
    with path.open("rb") as source:
        header_size = int.from_bytes(source.read(8), "little")
        if not 0 < header_size <= 10_000_000:
            raise ValueError(f"Invalid tensor header length in {path}")
        metadata = json.loads(source.read(header_size))
    offsets = [item["data_offsets"] for key, item in metadata.items() if key != "__metadata__"]
    if not offsets or max(end for _, end in offsets) != path.stat().st_size - 8 - header_size:
        raise ValueError(f"Incomplete tensor payload in {path}")


def checkpoint_paths(output):
    return sorted(path for path in (output / "checkpoints").glob("step-*")
                  if path.name[5:].isdigit() and (path / "state.pt").is_file())


def inspect_checkpoint(path, variant, *, crc=False):
    safetensors_intact(path / "adapter/adapter_model.safetensors")
    adapter = read_json(path / "adapter/adapter_config.json")
    if adapter.get("peft_type") != "LORA":
        raise ValueError("Optimizer checkpoint adapter has a different type")
    if variant == "spatial":
        safetensors_intact(path / "spatial.safetensors")
    with zipfile.ZipFile(path / "state.pt") as archive:
        members = archive.namelist()
        if not any(item.endswith("/data.pkl") for item in members):
            raise ValueError("Optimizer snapshot has no state metadata")
        if crc and archive.testzip() is not None:
            raise ValueError("Optimizer snapshot CRC failed")
    return int(path.name[5:])


def final_result(output):
    if not (output / "summary.json").exists():
        return False
    config = read_json(output / "training_config.json")
    result = read_json(output / "summary.json")
    expected = config["examples_per_epoch"] * config["epochs"]
    updates = math.ceil(expected / config["accumulation"])
    if (result.get("updates"), result.get("supervised_examples"), result.get("stopped_at_budget")) != (
            updates, expected, False):
        raise ValueError("Final summary does not cover the frozen training schedule")
    val = csv_tail(output / "val_loss.csv")
    if (int(val.get("update", 0)), int(val.get("frames", 0)), int(val.get("actions", 0))) != (updates, 6500, 2401):
        raise ValueError("Final validation does not cover the frozen split")
    if any(not math.isfinite(float(val[key])) for key in
           ("loss", "token_accuracy", "exact_action_accuracy")):
        raise ValueError("Final validation contains a nonfinite metric")
    safetensors_intact(output / "adapter/adapter_model.safetensors")
    if config["variant"] == "spatial":
        safetensors_intact(output / "spatial.safetensors")
    return True


def recovery_reason(log, status):
    if AUTO_RECOVERY in log:
        return "automatic_recovery_already_attempted", False
    if HARD_FAILURE.search(log):
        return "nontransient_training_error", False
    if STORAGE_FAILURE.search(log):
        return "transient_storage_interruption", True
    if status in (None, 129, 137, 143):
        return "launcher_interrupted_without_final_result", True
    return "unclassified_training_exit", False


def choose_resume_cache(run, values, options):
    manifest = run / "archive_sha256_disk.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(records) != 1036 or len({row["name"] for row in records}) != len(records):
        raise ValueError("Archive inventory is incomplete")
    original = Path(values["--archive-cache"])
    fallback = Path((run / "disk_cache_path").read_text().strip())
    for cache in dict.fromkeys((original, fallback)):
        if all(Path(row["name"]).name == row["name"] and (cache / row["name"]).is_file()
               and (cache / row["name"]).stat().st_size == row["bytes"] for row in records):
            if all(sha256(cache / row["name"]) == row["sha256"] for row in records):
                options[options.index("--archive-cache") + 1] = str(cache)
                return cache
    raise ValueError("Neither archive cache reproduces the frozen SHA-256 inventory")


def event(folder, name, **data):
    record = dict(time=stamp(), event=name, **data)
    with (folder / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def resume_worker(work, run, lane):
    folder = run / "gpu-supervisor"
    output = run / f"train-{lane}"
    state = read_json(folder / "state.json")["lanes"][lane]
    status_path = run / f"{lane}.status"
    snapshot = Path(state["resume_snapshot"])
    gpu = LANES[lane][0]
    log = run / f"{lane}.log"
    options = state["training_options"].copy()
    values, _ = resolve_args([*options, "--checkpoint", str(work / "models/JarvisVLA-Qwen2-VL-7B")],
                             work, run, lane)
    try:
        if sha256(output / "training_config.json") != state["config_sha256"]:
            raise ValueError("Training settings changed after capture")
        cfg = read_json(output / "training_config.json")
        if sha256(Path(values["--dataset"])) != cfg["dataset_sha256"]:
            raise ValueError("Frozen training index changed")
        if checkpoint_paths(output)[-1] != snapshot:
            raise ValueError("New checkpoint appeared after recovery selection")
        step = inspect_checkpoint(snapshot, lane, crc=True)
        cache = choose_resume_cache(run, values, options)
        roles = process_roles(process_args(), work, run, lane)
        if roles["train"] or roles["wrapper"] or final_result(output):
            raise ValueError("Original training or launcher returned before recovery")
        if not recovery_reason(latest_text(log, 2_000_000), exit_status(status_path))[1]:
            raise ValueError("Original failure or automatic recovery changed after selection")
        if not available_lock(work, gpu) or not query_gpu(gpu)["idle"]:
            raise ValueError("The experiment GPU is no longer confirmed idle")
        if status_path.exists():
            if exit_status(status_path) in (None, 0):
                raise ValueError("Original training status changed before recovery")
            status_path.replace(folder / f"{lane}-before-recovery.status")
        event(folder, "resume_verified", variant=lane, gpu=gpu, checkpoint=step, cache=str(cache),
              observed_update=state["resume_observed_update"], log=str(log))
        env = dict(os.environ, WORK=str(work), MINEMEM_PROXY="", CUDA_VISIBLE_DEVICES="")
        with log.open("a") as target:
            target.write(f"\n[{stamp()}] Supervisor resumes at update={step}\n")
            target.flush()
            result = subprocess.run(["bash", str(work / "code/minemem/t1_exec.sh"), "train",
                                     str(gpu), *options, "--resume"],
                                    cwd=work, env=env, stdout=target, stderr=subprocess.STDOUT)
        code = result.returncode
        if code == 0 and not final_result(output):
            raise ValueError("Resumed trainer exited before a complete result")
        atomic(status_path, f"{code}\n")
        event(folder, "resume_exited", variant=lane, gpu=gpu, code=code, log=str(log))
        return code
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
        event(folder, "resume_blocked", variant=lane, gpu=gpu, detail=str(error), log=str(log))
        if not status_path.exists() and (folder / f"{lane}-before-recovery.status").exists():
            (folder / f"{lane}-before-recovery.status").replace(status_path)
        return 1


def requeue_evaluation(work, run, folder, lane, status, roles, state, recovered_dependency):
    eval_status = exit_status(run / f"{lane}_eval.status")
    restore_dependency = recovered_dependency and eval_status not in (None, 0)
    if (not (state.get("resume_snapshot") or restore_dependency) or status != 0
            or roles["evaluation"] or state.get("evaluation_requeued")):
        return
    if any(run.glob(f"eval-{lane}-*")) or eval_status == 0:
        return
    waiter = run / f"after-{LANES[lane][1]}.sh"
    if sha256(waiter) != state.get("waiter_sha256"):
        event(folder, "evaluation_waiter_changed", variant=lane)
        return
    status_path = run / f"{lane}_eval.status"
    if status_path.exists():
        status_path.replace(folder / f"{lane}-eval-before-recovery.status")
    state["evaluation_requeued"] = True
    with (folder / f"{lane}-eval-reattach.log").open("a") as log:
        process = subprocess.Popen(["bash", str(waiter)], cwd=work, start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    event(folder, "evaluation_waiter_restored", variant=lane, pid=process.pid,
          recovered_dependency=restore_dependency)


def observe(work, run, folder, lane, roles, gpu_metrics, state, save_state, *, allow_recovery=True,
            recovered_dependency=False):
    gpu = LANES[lane][0]
    output = run / f"train-{lane}"
    log = run / f"{lane}.log"
    status = exit_status(run / f"{lane}.status")
    loss, validation = csv_tail(output / "loss.csv"), csv_tail(output / "val_loss.csv")
    update, validated = int(loss.get("update", 0)), int(validation.get("update", 0))
    found = checkpoint_paths(output)
    checkpoint = int(found[-1].name[5:]) if found else None
    active = bool(roles["train"] or roles["wrapper"] or roles["recover"])
    detail = ""
    if roles["train"] and not state.get("training_options"):
        argv = roles["train"][0][1]
        _, state["training_options"] = resolve_args(argv, work, run, lane)
        state["config_sha256"] = sha256(output / "training_config.json")
        state["waiter_sha256"] = sha256(run / f"after-{LANES[lane][1]}.sh")
        state["resume_attempted"] = "--resume" in argv
        event(folder, "trainer_registered", variant=lane, gpu=gpu, update=update, checkpoint=checkpoint)
    if active:
        state["inactive_samples"] = 0
        state["last_seen_update"] = update
        phase = ("validating" if roles["train"] and update and update % 1000 == 0
                 and validated < update else "training_or_encoding" if roles["train"] else "launcher_active")
        if state.get("progress_key") != [phase, update, validated, checkpoint]:
            state["progress_since"] = time.time()
        state["progress_key"] = [phase, update, validated, checkpoint]
        limit = max(900.0, 3 * float(validation.get("seconds", 0)))
        if time.time() - state["progress_since"] > limit:
            detail = "optimizer progress unchanged beyond the observation window"
            phase += "_slow"
    elif final_result(output):
        phase = "training_complete"
        if allow_recovery:
            requeue_evaluation(work, run, folder, lane, status, roles, state, recovered_dependency)
    else:
        state["inactive_samples"] = state.get("inactive_samples", 0) + 1
        phase = "confirming_exit"
        if state["inactive_samples"] >= 2:
            history = latest_text(log, 2_000_000)
            reason, retry = recovery_reason(history, status)
            detail = reason
            phase = "stopped_needs_diagnosis"
            if state.get("resume_attempted"):
                detail = "automatic_recovery_budget_exhausted"
            elif not state.get("training_options") or not found:
                detail = "no_registered_training_or_complete_snapshot"
            elif allow_recovery and retry and status != 0:
                phase = "waiting_for_idle_gpu"
                if gpu_metrics["idle"] and available_lock(work, gpu):
                    inspect_checkpoint(found[-1], lane)
                    state.update(resume_attempted=True, resume_snapshot=str(found[-1]),
                                 resume_observed_update=max(update, state.get("last_seen_update", 0)))
                    save_state()
                    with (folder / f"{lane}-resume-worker.log").open("a") as stdout:
                        worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                                   "--work", str(work), "--run", str(run), "--resume-worker", lane],
                                                  cwd=work, start_new_session=True, stdin=subprocess.DEVNULL,
                                                  stdout=stdout, stderr=subprocess.STDOUT)
                    phase = "resume_worker_started"
                    event(folder, phase, variant=lane, gpu=gpu, pid=worker.pid, trigger=reason,
                          checkpoint=checkpoint, original_status=status, log=str(log),
                          observed_update=state["resume_observed_update"],
                          rollback_updates=max(0, state["resume_observed_update"] - checkpoint))
    evaluation_status = exit_status(run / f"{lane}_eval.status")
    details = dict(phase=phase, variant=lane, gpu=gpu, update=update, validation=validated,
                   checkpoint=checkpoint, training_status=status, evaluation_status=evaluation_status,
                   pids=[pid for pid, _ in roles["train"]], wrapper_pids=roles["wrapper"],
                   evaluation_pids=roles["evaluation"], detail=detail)
    event_key = [phase, detail, checkpoint, validated, status, evaluation_status]
    if state.get("last_event_key") != event_key:
        state["last_event_key"] = event_key
        event(folder, "lane_state", **details, log=str(log))
    return details


def supervise(work, run, interval, once):
    folder = run / "gpu-supervisor"
    folder.mkdir(exist_ok=True)
    with (folder / "monitor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = folder / "state.json"
        state = read_json(state_path) if state_path.exists() else {"lanes": {x: {} for x in LANES}}

        def save_state():
            atomic(state_path, json.dumps(state, indent=2) + "\n")

        def poll():
            active_processes = list(process_args())
            results = []
            for lane, (gpu, _) in LANES.items():
                metrics = query_gpu(gpu)
                roles = process_roles(active_processes, work, run, lane)
                saved = state["lanes"][lane]
                restored_no_memory = (lane == "spatial"
                                      and state["lanes"]["no_memory"].get("evaluation_requeued", False)
                                      and exit_status(run / "no_memory_eval.status") in (None, 0))
                try:
                    info = observe(work, run, folder, lane, roles, metrics, saved, save_state,
                                   allow_recovery=not once, recovered_dependency=restored_no_memory)
                except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
                    info = dict(variant=lane, gpu=gpu, phase="monitor_preflight_error", detail=str(error))
                    if saved.get("monitor_error") != info["detail"]:
                        event(folder, "monitor_preflight_error", **info)
                    saved["monitor_error"] = info["detail"]
                results.append(dict(time=stamp(), **info, **metrics))
            atomic(folder / "latest.json", json.dumps(results, indent=2) + "\n")
            with (folder / "samples.jsonl").open("a") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in results))
            save_state()
            return results

        event(folder, "supervisor_started", pid=os.getpid(), interval=interval, once=once)
        while True:
            results = poll()
            if all(result["phase"] == "training_complete" for result in results):
                event(folder, "both_training_runs_complete")
                return 0
            if once:
                print(json.dumps(results, indent=2))
                return 0
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--resume-worker", choices=tuple(LANES))
    args = parser.parse_args()
    work, run = args.work.resolve(), args.run.resolve()
    if run != work / "results/phase3-speed-xOn8Q5je" or not run.is_dir():
        parser.error("Only the authorized screened Phase 3 output is monitored")
    if not 5 <= args.interval <= 60:
        parser.error("Choose a polling interval from 5 through 60 seconds")
    if args.resume_worker:
        return resume_worker(work, run, args.resume_worker)
    return supervise(work, run, args.interval, args.once)


if __name__ == "__main__":
    raise SystemExit(main())

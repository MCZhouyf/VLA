"""Repeat T1 acceptance and pilots with verified respawn and per-step telemetry."""

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path

from phase1_t1_eval import (
    FIELDS, Jarvis, MineStudioPolicy, RandomPolicy, fixture_index, return_segment,
)
from t1_safety import SETUP_REVISION, ProtectedArena, StepTrace, TrialInterrupted, total_flags
from t1_v4 import DISTANCES, PROTOCOL, RandomHeadingArena, assigned_seeds, horizontal, wrapped_angle


RADIUS = 3.0
GEOMETRY = ("arena_origin", "landmark", "distractors", "wall", "waypoints", "planned_endpoint")
PILOT_FIELDS = (
    *FIELDS, "deaths", "pose_jump", "pose_jumps", "end_reason", "setup_revision", "trace", "config",
)
ACCEPTANCE_FIELDS = (
    "d", "seed", "world_seed", "return_limit", "teacher_success", "teacher_steps",
    "teacher_final_dist", "teacher_min_dist", "teacher_deaths", "teacher_pose_jump",
    "teacher_pose_jumps", "teacher_end", "shortcut_success", "shortcut_steps",
    "shortcut_min_dist", "shortcut_deaths", "shortcut_pose_jump", "shortcut_pose_jumps",
    "shortcut_end", "setup_revision", "error",
)


def verify_scene(config, fixture):
    if config["protocol"] != PROTOCOL or config["world_seed"] != fixture["world_seed"]:
        raise RuntimeError("T1 protocol or world seed differs from the frozen fixture")
    for key in GEOMETRY:
        if json.loads(json.dumps(config[key])) != fixture[key]:
            raise RuntimeError(f"Frozen T1 geometry changed: {key}")


def freeze_return(config, fixture):
    if abs(wrapped_angle(config["final_yaw"] - fixture["final_yaw"])) > 2:
        raise RuntimeError("T1 final direction differs from the frozen fixture")
    if abs(config["actual_distance"] - fixture["actual_distance"]) > 0.65:
        raise RuntimeError("T1 return distance differs from the frozen fixture")
    config["return_pose"] = fixture["return_pose"]
    config["return_limit"] = fixture["return_limit"]


def record_config(folder, key, config):
    name = f"{key}.json"
    (folder / name).write_text(json.dumps(config, indent=2, default=float) + "\n")
    return name


def shortcut(arena, fixture, trace, protected=True):
    arena.restore_return_pose(fixture)
    initial = arena.info["player_pos"]["yaw"]
    closest = math.dist(horizontal(arena.info), arena.landmark)
    action = arena.sim.noop_action()
    action["forward"] = 1
    status, steps = "budget", 0
    try:
        with arena.track(trace) if protected else nullcontext():
            steps = arena.turn_to(wrapped_angle(initial + 180))
            if not protected:
                trace.observe(arena.info)
            for steps in range(steps+1, fixture["return_limit"]+1):
                if protected:
                    valid = arena.navigation_step(action.copy())
                else:
                    _, _, ended, truncated, arena.info = arena.sim.step(action.copy())
                    valid = trace.observe(arena.info, ended, truncated)
                if not valid:
                    status = trace.reason
                    break
                closest = min(closest, math.dist(horizontal(arena.info), arena.landmark))
                if closest <= RADIUS:
                    status = "reached"
                    break
    except TrialInterrupted as error:
        status = str(error)
        steps = len(trace.rows)-1
    finally:
        if not protected:
            trace.write()
    return status == "reached", steps, closest, *trace.flags(), status


def teacher(arena, config, fixture, folder, key):
    outbound_trace = StepTrace(folder / f"{key}-outbound.csv.gz")
    routes = []
    try:
        with arena.track(outbound_trace):
            arena.outbound_segment(config)
        freeze_return(config, fixture)
        arena.restore_return_pose(config)
        arena.teacher_traces = folder, key, routes
        ok, steps, final, closest = arena.teacher_return(config)
        # Preserve the evaluation budget even when the rerun teacher times differ.
        config["return_limit"] = fixture["return_limit"]
        return ok, steps, final, closest, *total_flags([outbound_trace, *routes]), (
            "reached" if ok else "teacher_budget"
        )
    except TrialInterrupted as error:
        measured = routes[-1] if routes else outbound_trace
        points = [math.dist((r[1], r[3]), arena.landmark) for r in measured.rows[:-1]]
        final = points[-1] if points else math.dist(horizontal(arena.info), arena.landmark)
        return (False, len(measured.rows)-1 if routes else 0, final,
                min(points, default=final), *total_flags([outbound_trace, *routes]), str(error))
    finally:
        arena.teacher_traces = None


def acceptance(args):
    fixtures = fixture_index(args.fixtures)
    wanted = {(int(d), int(seed)) for d, seed in args.case} if args.case else fixtures.keys()
    if not wanted <= fixtures.keys():
        raise ValueError("Selected acceptance cases must exist in the frozen fixture set")
    if args.skip_verified:
        for source in args.skip_verified.glob("worker-*/acceptance.csv"):
            with source.open(newline="") as stream:
                wanted = wanted - {(int(r["d"]), int(r["seed"])) for r in csv.DictReader(stream)
                                   if not r["error"]}
    with (args.output / "acceptance.csv").open("x", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(ACCEPTANCE_FIELDS)
        out.flush()
        for world, seeds in assigned_seeds(args.worker, args.workers):
            cases = [(d, seed) for seed in seeds for d in DISTANCES if (d, seed) in wanted]
            if not cases:
                continue
            arena = ProtectedArena(world, effects=args.effects)
            try:
                for d, seed in cases:
                    key = f"d{d}-seed{seed}"
                    fixture = fixtures[d, seed]
                    try:
                        config = arena.prepare_episode(seed, d)
                        verify_scene(config, fixture)
                        teach = teacher(arena, config, fixture, args.output, key)
                        config["teacher_result"] = dict(zip(ACCEPTANCE_FIELDS[4:12], teach, strict=True))
                        config["return_limit"] = fixture["return_limit"]
                        record_config(args.output, key, config)
                        direct = shortcut(arena, fixture, StepTrace(args.output / f"{key}-shortcut.csv.gz"))
                        row = (d, seed, world, fixture["return_limit"], *teach, *direct, SETUP_REVISION, "")
                    except (RuntimeError, OSError) as error:
                        row = (d, seed, world, fixture["return_limit"],
                               *("",) * (len(ACCEPTANCE_FIELDS)-6), SETUP_REVISION, str(error))
                    writer.writerow(row)
                    out.flush()
                    print(row, flush=True)
            finally:
                arena.sim.close()


class HistoricalArena(RandomHeadingArena):
    def __init__(self, world):
        self.trace = None
        super().__init__(world)

    def navigation_step(self, action):
        if not self.trace.rows:
            self.trace.observe(self.info)
        valid = super().navigation_step(action)
        self.trace.observe(self.info, terminated=not valid and self.navigation_end == "terminated")
        return valid


def policy_return(arena, policy, config, seed, limit, trace, historical):
    def observed_step(action):
        if arena.trace is None:
            # The explicit reset to the frozen pose is outside scored control.
            arena.trace = trace
        if historical:
            return arena.navigation_step(action)
        if not trace.rows and not trace.observe(arena.info):
            return False
        return arena.navigation_step(action)

    try:
        return return_segment(arena, policy, config, seed, limit, navigation_step=observed_step)
    finally:
        arena.trace = None
        trace.write()


def pilot(args):
    from t1_respawn_report import acceptance_summary

    if not args.historical and not all(row["accepted"] for row in acceptance_summary(args.fixtures)):
        raise SystemExit("T1 respawn acceptance has not passed")
    frozen = json.loads(args.seeds_file.read_text())
    if args.seed not in frozen["pilot"] or len(set(args.distances)) != len(args.distances):
        raise ValueError("Use a frozen pilot seed and distinct T1 distances")
    if "jarvis7b" in args.models and not args.base_url:
        raise ValueError("Provide the local JARVIS-VLA endpoint")
    fixtures = fixture_index(args.fixtures)
    world = frozen["world_seed_by_scenario"][str(args.seed)]
    policies = {}
    for model in args.models:
        if model == "jarvis7b":
            policies[model] = Jarvis(args.work, args.base_url)
        elif model == "random":
            policies[model] = RandomPolicy()
        else:
            policies[model] = MineStudioPolicy(args.work, model, "cpu")
    requested_effects = fixtures[args.distances[0], args.seed].get("optional_effects", False)
    arena = HistoricalArena(world) if args.historical else ProtectedArena(world, requested_effects)
    revision = "archived_v4_replay" if args.historical else SETUP_REVISION
    try:
        with (args.output / "cliff_T1.csv").open("x", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(PILOT_FIELDS)
            out.flush()
            for d in args.distances:
                fixture = fixtures[d, args.seed]
                key = f"d{d}-seed{args.seed}"
                config = arena.prepare_episode(args.seed, d)
                verify_scene(config, fixture)
                outbound_trace = StepTrace(args.output / f"{key}-outbound.csv.gz")
                outbound_failure = ""
                try:
                    with nullcontext() if args.historical else arena.track(outbound_trace):
                        arena.outbound_segment(config)
                    freeze_return(config, fixture)
                    if args.historical:
                        with (args.output / f"{key}-outbound.csv").open("x", newline="") as stream:
                            writer_out = csv.writer(stream)
                            writer_out.writerow(("x", "z", "yaw", "action"))
                            writer_out.writerows(arena.outbound)
                except TrialInterrupted as error:
                    outbound_failure = str(error)
                config["setup_revision"] = revision
                config_name = record_config(args.output, key, config)
                arena.command("/gamemode adventure @p")
                arena.settle()
                for model, policy in policies.items():
                    trace_name = f"{key}-{model}-return.csv.gz"
                    trace = StepTrace(args.output / trace_name)
                    if outbound_failure:
                        ok, steps, final, closest, elapsed = False, 0, d, d, 0.0
                        flags, reason = outbound_trace.flags(), f"outbound_{outbound_failure}"
                        trace_name = ""
                    else:
                        ok, steps, final, closest, elapsed = policy_return(
                            arena, policy, config, args.seed, fixture["return_limit"], trace, args.historical)
                        flags = trace.flags()
                        reason = trace.reason or ("reached" if ok else "budget")
                    writer.writerow((model, d, args.seed, ok, steps, final, closest, arena.path_len_out,
                                     world, fixture["return_limit"], elapsed, *flags, reason, revision,
                                     trace_name, config_name))
                    out.flush()
                    print(model, d, args.seed, ok, steps, closest, flags, reason, flush=True)
                arena.command("/gamemode survival @p")
    finally:
        arena.sim.close()


def death_replay(args):
    configs = fixture_index(args.fixtures)
    fixture = configs[args.distances[0], args.seed]
    arena = RandomHeadingArena(fixture["world_seed"])
    try:
        config = arena.prepare_episode(args.seed, fixture["distance"])
        verify_scene(config, fixture)
        trace = StepTrace(args.output / "shortcut.csv.gz")
        result = shortcut(arena, fixture, trace, protected=False)
        record_config(args.output, "summary", dict(
            seed=args.seed, distance=fixture["distance"], old_respawn=arena.native_spawn,
            success=result[0], steps=result[1], min_dist=result[2], deaths=result[3],
            pose_jump=result[4], pose_jumps=result[5], end_reason=result[6],
        ))
    finally:
        arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    check = modes.add_parser("acceptance")
    check.add_argument("output", type=Path)
    check.add_argument("--fixtures", type=Path, nargs="+", required=True)
    check.add_argument("--worker", type=int, required=True)
    check.add_argument("--workers", type=int, required=True)
    check.add_argument("--case", action="append", nargs=2, metavar=("DISTANCE", "SEED"))
    check.add_argument("--skip-verified", type=Path)
    check.add_argument("--effects", action="store_true")
    run = modes.add_parser("pilot")
    run.add_argument("output", type=Path)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--fixtures", type=Path, nargs="+", required=True)
    run.add_argument("--seeds-file", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--distances", type=int, choices=DISTANCES, nargs="+", default=DISTANCES)
    run.add_argument("--models", choices=("random", "steve1", "vpt", "jarvis7b"), nargs="+", required=True)
    run.add_argument("--base-url")
    run.add_argument("--historical", action="store_true")
    death = modes.add_parser("death-replay")
    death.add_argument("output", type=Path)
    death.add_argument("--fixtures", type=Path, nargs="+", required=True)
    death.add_argument("--seed", type=int, required=True)
    death.add_argument("--distances", type=int, choices=DISTANCES, nargs=1, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"acceptance": acceptance, "pilot": pilot, "death-replay": death_replay}[args.mode](args)


if __name__ == "__main__":
    main()

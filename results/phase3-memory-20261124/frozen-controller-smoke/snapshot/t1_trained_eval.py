"""Evaluate trained T1 policies in MineStudio using the frozen protocol."""

import argparse
import csv
import io
import json
import math
from collections import deque
from contextlib import ExitStack
from copy import copy
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from phase1_t1_eval import fixture_index
from t1_collect import (
    NAV_SEED, NAV_WORLD_SEED, ProtectedCaptureArena, RevisedCaptureArena, navigation_cases,
)
from t1_model import T1Policy
from t1_occlusion import SCREENED_PROTOCOL, set_candidate_render_distance
from t1_respawn import GEOMETRY, freeze_return, record_config, verify_scene
from t1_respawn_report import acceptance_summary
from t1_safety import SETUP_REVISION, StepTrace, TrialInterrupted
from t1_v4 import (
    DISTANCES, INSTRUCTION, SUCCESS_RADIUS, VISIBLE_INSTRUCTION, horizontal,
)


class Controller:
    def __init__(self, checkpoint, training_run, device="cuda", cache_vision=False):
        self.settings = json.loads((training_run / "training_config.json").read_text())
        self.cache_vision = cache_vision or self.settings.get("cache_vision", False)
        self.policy = T1Policy(checkpoint, self.settings["variant"],
                               quantized=self.settings["quantized"], adapter=training_run / "adapter",
                               device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
        if self.policy.memory is not None:
            from safetensors.torch import load_file

            self.policy.memory.load_state_dict(load_file(training_run / "spatial.safetensors"))
            self.policy.memory.position_value_scale = self.settings.get("position_value_scale", 0.0)
        self.policy.eval()
        self.control_trace = None

    def image(self, frame):
        resolution = self.settings.get("resolution")
        if resolution:
            frame = frame.resize(tuple(resolution), Image.Resampling.BILINEAR)
        return self.policy.images.create_image_input(frame)

    def start(self, outbound):
        images = []
        actions = []
        poses = []
        if self.policy.memory is None:
            outbound = outbound[-self.policy.history:]
        for jpeg, pose, action in outbound:
            with Image.open(io.BytesIO(jpeg)) as frame:
                images.append(self.image(frame.convert("RGB")))
            poses.append(pose)
            actions.append(action)
        self.poses = poses
        self.features = []
        if self.cache_vision:
            encoded = []
            for begin in range(0, len(images), 3):
                encoded.extend(self.policy.encode_frames(images[begin:begin + 3]))
            images = encoded
        self.history = deque(zip(images, actions), maxlen=self.policy.history)
        if self.policy.memory is not None:
            for begin in range(0, len(images), 8):
                self.features.extend(self.policy.pooled_images(images[begin:begin + 8]))

    def request(self, info, instruction, image):
        pose = [float(info["player_pos"][key]) for key in ("x", "z", "yaw")]
        frames = [frame for frame, _ in self.history] + [image]
        actions = [action for _, action in self.history]
        if len(actions) < self.policy.history:
            missing = self.policy.history - len(actions)
            frames = [frames[0]] * missing + frames
            actions = [self.policy.actions.null_token()] * missing + actions
        if self.settings.get("evaluation_history") == "placeholder":
            actions = [self.policy.actions.null_token()] * self.policy.history
        features = poses = None
        if self.policy.memory is not None:
            features = torch.stack(self.features) if self.features else torch.empty(
                (0, self.policy.base.config.hidden_size))
            poses = [*self.poses, pose]
        return (instruction, frames, actions, features, poses), pose

    def record_action(self, image, pose, prediction):
        control, generated, tags = prediction
        self.history.append((image, generated))
        if self.policy.memory is not None:
            self.poses.append(pose)
            self.features.extend(self.policy.pooled_images([image]))
        return control, tags

    def action(self, info, instruction):
        return choose_actions([(self, info, instruction)])[0]


def choose_actions(requests):
    controller, _, _ = requests[0]
    policy = controller.policy
    images = [item.image(Image.fromarray(info["pov"])) for item, info, _ in requests]
    if controller.cache_vision:
        images = policy.encode_frames(images)
    prepared, poses = zip(*(item.request(info, instruction, image) for (item, info, instruction), image
                            in zip(requests, images, strict=True)), strict=True)
    capture = any(item.control_trace is not None for item, _, _ in requests)
    predictions = (policy.predict_batch(prepared, return_ids=capture) if len(prepared) > 1 else
                   [policy.predict(*prepared[0], return_ids=capture)])
    result = []
    for (item, _, _), image, pose, prediction, example in zip(
        requests, images, poses, predictions, prepared, strict=True
    ):
        if item.control_trace is not None:
            agent, text, complete, ids = prediction
            item.control_trace.record_prediction(example[2], agent, text, complete, ids)
        result.append(item.record_action(image, pose, prediction[:3]))
    return result


class ControlTrace(StepTrace):
    def __init__(self, path, control_path):
        super().__init__(path)
        self.control_path = control_path
        self.controls = []

    def record_prediction(self, history, agent, text, complete, ids):
        self.controls.append(dict(
            step=len(self.controls), history_actions=history, raw_token_ids=ids, raw_text=text,
            complete=bool(complete), agent_buttons=int(agent["buttons"]), agent_camera=int(agent["camera"]),
        ))

    def record_environment(self, action):
        pitch, yaw = action["camera"]
        self.controls[-1].update(
            pitch=float(pitch), yaw=float(yaw),
            buttons={key: int(value) for key, value in action.items() if key != "camera"},
        )

    def write(self):
        super().write()
        with self.control_path.open("x") as stream:
            for row in self.controls:
                stream.write(json.dumps(row) + "\n")


def rollout(arena, controller, config, outbound, instruction, limit, trace):
    arena.restore_return_pose(config)
    controller.start(outbound)
    closest = math.dist(horizontal(arena.info), arena.landmark)
    tags = 0
    final = closest
    steps = 0
    start = perf_counter()
    if isinstance(trace, ControlTrace):
        controller.control_trace = trace
    try:
        with arena.track(trace):
            for steps in range(1, limit + 1):
                action, complete = yield controller, arena.info, instruction
                tags += complete
                action = arena.sim.agent_action_to_env_action(action)
                if controller.control_trace is not None:
                    trace.record_environment(action)
                if not arena.navigation_step(action):
                    break
                final = math.dist(horizontal(arena.info), arena.landmark)
                closest = min(closest, final)
                if closest <= SUCCESS_RADIUS:
                    break
    except TrialInterrupted:
        pass
    finally:
        controller.control_trace = None
    ok = closest <= SUCCESS_RADIUS and not trace.reason
    return (ok, steps, final, closest, tags, perf_counter() - start,
            *trace.flags(), trace.reason or ("reached" if ok else "budget"))


def drive_rollouts(rollouts, on_done):
    active = {}
    with ExitStack() as cleanup:
        for index, lane in enumerate(rollouts):
            cleanup.callback(lane.close)
            try:
                active[index] = next(lane)
            except StopIteration as done:
                on_done(index, done.value)
        while active:
            indices = list(active)
            decisions = choose_actions([active[index] for index in indices])
            for index, decision in zip(indices, decisions, strict=True):
                try:
                    active[index] = rollouts[index].send(decision)
                except StopIteration as done:
                    del active[index]
                    on_done(index, done.value)


def run_case(arena, controller, config, outbound, instruction, limit, trace):
    results = []
    drive_rollouts([rollout(arena, controller, config, outbound, instruction, limit, trace)],
                   lambda _, result: results.append(result))
    return results[0]


def screened_rollout(args, seed, distance, expected, controller):
    arena = RevisedCaptureArena(expected["world_seed"], expected["optional_effects"])
    try:
        config = arena.prepare_episode(seed, distance)
        verify_screened_scene(config, expected)
        key = f"d{distance}-seed{seed}"
        limit = expected["return_limit"]
        if expected.get("outbound_error"):
            record_config(args.output, key, expected)
            return ((False, 0, distance, distance, 0, 0, 0, False, 0,
                     "outbound_frozen_teacher_failure"), limit, "")
        trace = StepTrace(args.output / f"{key}-outbound.csv.gz")
        arena.capture = []
        try:
            with arena.track(trace):
                arena.outbound_segment(config)
            outbound = arena.capture
        except TrialInterrupted:
            result = (False, 0, distance, distance, 0, 0, *trace.flags(), f"outbound_{trace.reason}")
            config["path_len_out"] = arena.path_len_out
        finally:
            arena.capture = None
        if not trace.reason:
            freeze_return(config, expected)
            arena.command("/gamemode adventure @p")
            arena.settle()
            trace = StepTrace(args.output / f"{key}-return.csv.gz")
            result = yield from rollout(arena, controller, config, outbound, INSTRUCTION, limit, trace)
        record_config(args.output, key, config)
        return result, limit, config["path_len_out"]
    finally:
        arena.sim.close()


def gate_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def validate_offset_gate(path, training_run, forward_control=None, variant="no_memory"):
    rows = gate_rows(path)
    expected = {seed:(distance,heading,cohort) for seed,distance,heading,cohort in navigation_cases(True)}
    if len(rows) != 20 or {int(row["seed"]) for row in rows} != expected.keys():
        raise ValueError("Offset navigation must cover its 20 frozen scenarios")
    visible = 0
    for row in rows:
        distance, heading, cohort = expected[int(row["seed"])]
        safe = float(row["min_dist"]) <= SUCCESS_RADIUS and int(row["deaths"]) == 0 and row["pose_jump"] == "False"
        if (row["model"] != variant or row["mode"] != "nav_offset"
                or row["cohort"] != cohort or abs(float(row["d"])-distance) > 1e-8
                or abs(float(row["heading_offset"])-heading) > 1e-8 or row["training_run"] != training_run
                or row["setup_revision"] != SETUP_REVISION or row["success"] != str(safe)):
            raise ValueError("Offset navigation result does not match its frozen scenario")
        visible += int(cohort == "in_view" and safe)
    if visible < 12:
        raise ValueError(f"Offset navigation reached {visible}/15 in-view scenes, below 12/15")
    if forward_control is not None:
        reference = [row for row in gate_rows(forward_control) if row["gate"] == "offset"]
        if len(reference) != 20 or {int(row["seed"]) for row in reference} != expected.keys():
            raise ValueError("Constant-forward results must cover the same offset scenarios")
        reference_visible = sum(row["cohort"] == "in_view" and row["success"] == "True"
                                for row in reference)
        if visible - reference_visible < 3:
            raise ValueError("Offset navigation must outperform constant forward by three in-view scenes")


def frozen_inputs(args):
    frozen = json.loads(args.seeds_file.read_text())
    if not args.seed or not set(args.seed) <= set(frozen["formal"]) or len(set(args.seed)) != len(args.seed):
        raise ValueError("Select distinct frozen formal scenario seeds")
    return frozen


def screened_inputs(args, frozen):
    screened = json.loads(args.screened_fixtures.read_text())
    if screened["formal"] != frozen["formal"] or screened["protocol"] != SCREENED_PROTOCOL:
        raise ValueError("Screened fixtures do not match the formal seed list")
    scenes = {(item["distance"], item["scenario_seed"]): item
              for item in screened["fixtures"].values()}
    if scenes.keys() != {(d, seed) for d in DISTANCES for seed in frozen["formal"]}:
        raise ValueError("Screened fixtures lack full formal coverage")
    return scenes


def diagnostic_inputs(args):
    if not args.seeds_file or not args.screened_fixtures or args.distances != [30]:
        raise ValueError("Diagnostic evaluation requires screened fixtures at 30 blocks")
    frozen = frozen_inputs(args)
    if set(args.seed) != set(frozen["formal"]):
        raise ValueError("Diagnostic evaluation requires all frozen scenario seeds")
    return frozen, screened_inputs(args, frozen)


def validate_aligned_gate(path, variant):
    nav = gate_rows(path)
    if ({int(row["seed"]) for row in nav} != set(range(NAV_SEED, NAV_SEED + 10))
            or len(nav) != 10
            or any(row["model"] != variant or row["mode"] != "nav"
                   or int(row["d"]) != 5
                   or row.get("setup_revision") != SETUP_REVISION
                   or row["success"] != str(float(row["min_dist"]) <= SUCCESS_RADIUS
                                            and int(row["deaths"]) == 0
                                            and row["pose_jump"] == "False")
                   for row in nav)
            or sum(row["success"] == "True" for row in nav) < 8):
        raise SystemExit(f"The {variant.replace('_', '-')} model did not pass visible navigation at 8/10")
    gate_runs = {row["training_run"] for row in nav}
    if len(gate_runs) != 1:
        raise SystemExit("Each navigation gate must come from one training run")
    return gate_runs.pop()


def formal_inputs(args, settings, training_run):
    if not args.no_memory_gate or not args.seeds_file or not (args.acceptance or args.screened_fixtures):
        raise SystemExit("Formal evaluation requires the passed no-memory navigation gate and T1 fixtures")
    gate_run = validate_aligned_gate(args.no_memory_gate, "no_memory")
    if settings["variant"] == "no_memory" and gate_run != training_run:
        raise SystemExit("The gate must come from the no-memory run being evaluated")
    frozen = frozen_inputs(args)
    if args.screened_fixtures:
        if not all((args.offset_gate, args.forward_control, args.spatial_gate, args.spatial_offset_gate)):
            raise ValueError("Screened formal evaluation requires both variants' navigation gates")
        validate_offset_gate(args.offset_gate, gate_run, args.forward_control)
        spatial_run = validate_aligned_gate(args.spatial_gate, "spatial")
        if settings["variant"] == "spatial" and spatial_run != training_run:
            raise SystemExit("The gate must come from the spatial run being evaluated")
        validate_offset_gate(args.spatial_offset_gate, spatial_run, args.forward_control, "spatial")
        return frozen, screened_inputs(args, frozen)
    if not all(row["accepted"] for row in acceptance_summary(args.acceptance)):
        raise SystemExit("T1 respawn teacher and shortcut gate did not pass")
    requirements = args.training_run.parent / "review_requirements.json"
    if requirements.is_file():
        review = json.loads(requirements.read_text())
        audit = Path(review["visibility_summary"])
        if not audit.is_file():
            raise ValueError("Frozen visibility and RGB scan review is still pending")
        shortcut = json.loads(audit.read_text())
        if any(shortcut[str(d)]["n"] != 30 or shortcut[str(d)]["scan_rate"] >= .10 for d in (30,60,100)):
            raise ValueError("The frozen scenes did not pass the RGB scan control")
        validate_offset_gate(Path(review["offset_gate"]), gate_run)
        decision = Path(review["early_decision"])
        if (settings["variant"] != "no_memory" or tuple(args.distances) != (30,)) and (
            not decision.is_file() or not json.loads(decision.read_text())["proceed"]
        ):
            raise ValueError("The no-memory 30-block comparison has not cleared the early stop")
    return frozen, fixture_index(args.acceptance)


def verify_screened_scene(config, expected):
    keys = (*GEOMETRY, "protocol", "world_seed", "scenario_seed", "occlusion_walls",
            "pillar_screen", "screened_pillar", "render_distance_chunks")
    for key in keys:
        if json.loads(json.dumps(config.get(key))) != expected.get(key):
            raise RuntimeError(f"Frozen screened geometry changed: {key}")


def run(args):
    training_run = str(args.training_run.resolve())
    settings = json.loads((args.training_run / "training_config.json").read_text())
    if args.mode == "formal":
        frozen, fixtures = formal_inputs(args, settings, training_run)
    elif args.mode == "diagnostic_d30":
        frozen, fixtures = diagnostic_inputs(args)
    screened_rollout_mode = args.mode in ("formal", "diagnostic_d30")
    if args.screened_fixtures and screened_rollout_mode:
        set_candidate_render_distance(args.output)
    controller = Controller(args.checkpoint, args.training_run, args.device, args.cache_vision)
    variant = controller.settings["variant"]
    with (args.output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "model", "seed", "d", "world_seed", "success", "steps",
                         "final_dist", "min_dist", "decoded_steps", "elapsed_seconds",
                         "deaths", "pose_jump", "pose_jumps", "end_reason", "return_limit",
                         "path_len_out", "training_run", "setup_revision", "cohort",
                         "heading_offset", "inference_device", "inference_dtype", "scene_protocol"))

        def output_row(seed, distance, world, result, limit, length, cohort="", offset=""):
            writer.writerow((args.mode, variant, seed, distance, world, *result,
                             limit, length, training_run, SETUP_REVISION, cohort, offset,
                             args.device, str(controller.policy.base.dtype),
                             SCREENED_PROTOCOL if args.screened_fixtures else "t1_v4"))
            stream.flush()

        if not screened_rollout_mode:
            arena = ProtectedCaptureArena(NAV_WORLD_SEED)
            try:
                limit = 200 if args.mode == "nav" else 30
                if args.mode == "nav_offset":
                    limit = 200
                cases = navigation_cases(args.mode == "nav_offset")
                if args.seed:
                    if len(set(args.seed)) != len(args.seed) or not set(args.seed) <= {case[0] for case in cases}:
                        raise ValueError("Select distinct seeds from the selected navigation cohort")
                    cases = [case for case in cases if case[0] in args.seed]
                for seed, distance, heading, cohort in cases[:1] if args.mode == "smoke" else cases:
                    config, outbound, teacher_return = arena.visible_trajectory(seed, distance, heading)
                    config["return_limit"] = limit
                    key = f"d5-seed{seed}" if args.mode != "nav_offset" else f"offset-seed{seed}"
                    record_config(args.output, key, config)
                    if args.start_probes:
                        from t1_control_probe import inspect_start

                        inspect_start(args.output, arena, controller, config, outbound, teacher_return, cohort)
                        continue
                    pose_path = args.output / f"{key}-return.csv.gz"
                    trace = (ControlTrace(pose_path, args.output / f"{key}-control.jsonl")
                             if args.control_trace else StepTrace(pose_path))
                    result = run_case(arena, controller, config, outbound, VISIBLE_INSTRUCTION, limit, trace)
                    output_row(seed, distance, NAV_WORLD_SEED, result, limit, "", cohort, heading)
            finally:
                arena.sim.close()
            return
        if args.screened_fixtures and args.rollout_batch_size > 1:
            cases = [(seed, distance) for distance in args.distances for seed in args.seed]
            with (args.output / "rollout_timing.csv").open("x", newline="") as log:
                timing = csv.writer(log)
                timing.writerow(("episodes", "return_steps", "wall_seconds", "aggregate_return_fps"))
                for begin in range(0, len(cases), args.rollout_batch_size):
                    batch = cases[begin:begin + args.rollout_batch_size]
                    lanes = [screened_rollout(args, seed, distance, fixtures[distance, seed], copy(controller))
                             for seed, distance in batch]
                    steps = 0

                    def output_lane(index, payload):
                        nonlocal steps
                        seed, distance = batch[index]
                        world = fixtures[distance, seed]["world_seed"]
                        steps += payload[0][1]
                        output_row(seed, distance, world, *payload)

                    started = perf_counter()
                    drive_rollouts(lanes, output_lane)
                    elapsed = perf_counter() - started
                    timing.writerow((len(batch), steps, elapsed, steps / elapsed))
                    log.flush()
            return
        for seed in args.seed:
            world = frozen["world_seed_by_scenario"][str(seed)]
            effects = fixtures[DISTANCES[0], seed]["optional_effects"]
            arena_type = RevisedCaptureArena if args.screened_fixtures else ProtectedCaptureArena
            arena = arena_type(world, effects)
            try:
                for distance in args.distances:
                    expected = fixtures[distance, seed]
                    config = arena.prepare_episode(seed, distance)
                    if args.screened_fixtures:
                        verify_screened_scene(config, expected)
                    else:
                        verify_scene(config, expected)
                    key = f"d{distance}-seed{seed}"
                    if expected.get("outbound_error"):
                        result = (False, 0, distance, distance, 0, 0, 0, False, 0,
                                  "outbound_frozen_teacher_failure")
                        record_config(args.output, key, expected)
                        output_row(seed, distance, world, result, expected["return_limit"], "")
                        continue
                    outbound_trace = StepTrace(args.output / f"{key}-outbound.csv.gz")
                    arena.capture = []
                    try:
                        with arena.track(outbound_trace):
                            arena.outbound_segment(config)
                        outbound = arena.capture
                    except TrialInterrupted:
                        result = (False, 0, distance, distance, 0, 0, *outbound_trace.flags(),
                                  f"outbound_{outbound_trace.reason}")
                        config["path_len_out"] = arena.path_len_out
                    finally:
                        arena.capture = None
                    if not outbound_trace.reason:
                        freeze_return(config, expected)
                        arena.command("/gamemode adventure @p")
                        arena.settle()
                        trace = StepTrace(args.output / f"{key}-return.csv.gz")
                        result = run_case(arena, controller, config, outbound, INSTRUCTION,
                                          expected["return_limit"], trace)
                        arena.command("/gamemode survival @p")
                    record_config(args.output, key, config)
                    output_row(seed, distance, world, result, expected["return_limit"], config["path_len_out"])
            finally:
                arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "nav", "nav_offset", "formal", "diagnostic_d30"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-vision", action="store_true")
    parser.add_argument("--control-trace", action="store_true")
    parser.add_argument("--start-probes", action="store_true")
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-memory-gate", type=Path)
    parser.add_argument("--offset-gate", type=Path)
    parser.add_argument("--spatial-gate", type=Path)
    parser.add_argument("--spatial-offset-gate", type=Path)
    parser.add_argument("--forward-control", type=Path)
    parser.add_argument("--screened-fixtures", type=Path)
    parser.add_argument("--acceptance", nargs="+", type=Path)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=DISTANCES)
    args = parser.parse_args()
    if args.rollout_batch_size < 1 or (args.rollout_batch_size > 1 and
                                      (args.mode not in ("formal", "diagnostic_d30") or not args.screened_fixtures)):
        parser.error("Multiple rollout lanes require a screened T1 evaluation")
    if args.start_probes and (args.mode not in ("nav", "nav_offset") or args.control_trace):
        parser.error("Start probes require a navigation cohort and use their own trace")
    if args.control_trace and args.mode not in ("nav", "nav_offset"):
        parser.error("Control traces require a navigation evaluation")
    run(args)


if __name__ == "__main__":
    main()

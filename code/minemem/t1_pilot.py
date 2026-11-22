"""Run the zero-shot T1 v4 reference policies on the five frozen pilot seeds."""

import argparse
import csv
import json
from pathlib import Path

from phase1_t1_eval import Jarvis, MineStudioPolicy, RandomPolicy, evaluate_fixture, fixture_index
from t1_v4 import DISTANCES, PROTOCOL, RandomHeadingArena, wrapped_angle
from t1_v4_report import summarize
from phase1_t1_report import MODELS, validate_results, summarize as summarize_models, write_csv, plot


class PilotArena(RandomHeadingArena):
    def __init__(self, world_seed, fixtures):
        self.fixtures = fixtures
        super().__init__(world_seed)

    def outbound_segment(self, config):
        super().outbound_segment(config)
        expected = self.fixtures[config["distance"], config["scenario_seed"]]
        if abs(wrapped_angle(config["final_yaw"] - expected["final_yaw"])) > 2:
            raise RuntimeError("T1 replay differs from frozen randomized orientation")
        # Every policy starts from the exact archived position and heading.
        config["return_pose"] = expected["return_pose"]


def run(args):
    if len(set(args.distances)) != len(args.distances):
        raise ValueError("Select distinct T1 distances")
    gates = summarize(args.acceptance)
    if not all(row["accepted"] for row in gates):
        raise SystemExit("Randomized-heading teacher and shortcut acceptance has not passed")
    frozen = json.loads(args.seeds_file.read_text())
    if args.seed not in frozen["pilot"]:
        raise ValueError("Only the frozen pilot seeds may be used here")
    fixtures = fixture_index(args.acceptance)
    world_seed = frozen["world_seed_by_scenario"][str(args.seed)]
    if "jarvis7b" in args.models and args.base_url is None:
        raise SystemExit("Provide the local JARVIS-VLA endpoint")
    policies = {}
    for model in args.models:
        if model == "jarvis7b":
            policy = Jarvis(args.work, args.base_url)
        elif model == "random":
            policy = RandomPolicy()
        else:
            policy = MineStudioPolicy(args.work, model, "cpu")
        policies[model] = policy
    args.output.mkdir(parents=True, exist_ok=True)
    arena = PilotArena(world_seed, fixtures)
    try:
        from phase1_t1_eval import FIELDS
        with (args.output / "cliff_T1.csv").open("x", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for d in args.distances:
                fixture = fixtures[d, args.seed]
                if fixture["protocol"] != PROTOCOL or fixture["world_seed"] != world_seed:
                    raise RuntimeError("Invalid pilot fixture")
                evaluate_fixture(arena, fixture, policies, writer, navigation_step=arena.navigation_step)
                output.flush()
    finally:
        arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("run")
    pilot.add_argument("--work", required=True, type=Path)
    pilot.add_argument("--output", required=True, type=Path)
    pilot.add_argument("--acceptance", required=True, type=Path, nargs="+")
    pilot.add_argument("--seeds-file", required=True, type=Path)
    pilot.add_argument("--seed", required=True, type=int)
    pilot.add_argument("--distances", type=int, choices=DISTANCES, nargs="+", default=DISTANCES)
    pilot.add_argument("--base-url")
    pilot.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    report = commands.add_parser("report")
    report.add_argument("output", type=Path)
    report.add_argument("--inputs", required=True, type=Path, nargs="+")
    report.add_argument("--seeds-file", required=True, type=Path)
    report.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return
    seeds = json.loads(args.seeds_file.read_text())["pilot"]
    if len(set(seeds)) != 5 or len(set(args.models)) != len(args.models):
        raise ValueError("Expected five distinct frozen seeds and distinct models")
    rows = validate_results(args.inputs, seeds, args.models)
    summary = summarize_models(rows, args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = "cliff_T1_v4_pilot" if set(args.models) == set(MODELS) else "cliff_T1_v4_pilot_partial"
    main_rows = [r for r in rows if r["model"] != "vpt"]
    appendix = [r for r in rows if r["model"] == "vpt"]
    if main_rows:
        write_csv(args.output / f"{stem}.csv", main_rows)
        plot(args.output / f"{stem}.svg", summary, monochrome=True)
    if appendix:
        write_csv(args.output / "cliff_T1_v4_vpt_appendix.csv", appendix)
    write_csv(args.output / f"{stem}_summary.csv", summary)


if __name__ == "__main__":
    main()

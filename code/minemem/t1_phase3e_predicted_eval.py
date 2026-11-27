"""Evaluate learned and grid-planned waypoints with the frozen Phase A controller."""

import argparse
import sys
from pathlib import Path
from time import perf_counter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "checkpoint", "training-run", "fixtures", "metrics", "offline-gate",
                 "oracle-gate", "room-weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("initial", "waypoint-weights"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--grid-plan", choices=("visual", "shuffle_visual"))
    parser.add_argument("--tracking-repair", action="store_true")
    parser.add_argument("--variant", choices=("no_memory", "long", "spatial"), required=True)
    parser.add_argument("--distances", type=int, nargs="+", choices=(10, 30, 60, 100), required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--no-stabilizer", action="store_true")
    args = parser.parse_args(argv)
    if args.batch < 1 or len(set(args.distances)) != len(args.distances):
        parser.error("Invalid rollout batch or duplicate distance")
    if args.grid_plan:
        if args.variant != "spatial" or args.no_stabilizer:
            parser.error("Grid planning requires the spatial arm and the shared stabilizer")
    else:
        if args.tracking_repair:
            parser.error("Route tracking repair requires --grid-plan")
        if args.initial is None or args.waypoint_weights is None:
            parser.error("Learned waypoints require --initial and --waypoint-weights")
    return args


def main():
    args = parse_args()
    start = perf_counter()
    print("Loading waypoint evaluator dependencies", file=sys.stderr, flush=True)
    from t1_phase3e_predicted_rollout import run

    print(f"Waypoint evaluator dependencies loaded in {perf_counter() - start:.1f}s",
          file=sys.stderr, flush=True)
    run(args)


if __name__ == "__main__":
    main()

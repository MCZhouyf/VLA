"""Stage 1 T1 arena, privileged teacher, and CPU acceptance runs."""

import argparse
import csv
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np


DISTANCES = (10, 30, 60, 100)
TEACHER_SEED = 2026112100
WORLD_SEED = 19961103
WORLD_COUNT = 32
# World 19961127 failed the arena air-block check before any teacher episode.
WORLD_REPLACEMENTS = {24: WORLD_SEED + WORLD_COUNT}
SUCCESS_RADIUS = 3.0
POINT_TOLERANCE = 0.45
CAMERA = dict(camera_binsize=1, camera_maxval=10, camera_mu=20,
              camera_quantization_scheme="mu_law")


def wrapped_angle(angle):
    return (angle + 180) % 360 - 180


def bearing(start, end):
    return math.degrees(math.atan2(start[0] - end[0], end[1] - start[1]))


def horizontal(info):
    pose = info["player_pos"]
    return float(pose["x"]), float(pose["z"])


class Arena:
    def __init__(self, world_seed):
        from jarvisvla.inference.action_mapping import OneActionTokenizer
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.entry import CameraConfig

        self.world_seed = world_seed
        self.tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
        self.sim = MinecraftSim(
            action_type="env", seed=world_seed, preferred_spawn_biome="plains",
            obs_size=(640, 360), render_size=(640, 360), camera_config=CameraConfig(**CAMERA),
        )
        self.scene_fills = []
        try:
            _, self.info = self.sim.reset()
            self.native_spawn = self.info["player_pos"]
            self.x, self.y, self.z = (math.floor(self.native_spawn[key])
                                      for key in ("x", "y", "z"))
            self.y += 12
            self.landmark = (self.x + 0.5, self.z + 0.5)
            self.prepare_ground()
        except BaseException:
            self.sim.close()
            raise

    def command(self, command):
        self.sim.env.execute_cmd(command)

    def tick(self, action=None):
        if action is None:
            action = self.sim.noop_action()
        _, _, terminated, truncated, self.info = self.sim.step(action)
        if terminated or truncated:
            raise RuntimeError("T1 simulator ended")

    def settle(self, steps=6):
        for _ in range(steps):
            self.tick()

    def fill(self, x1, y1, z1, x2, y2, z2, material):
        command = f"/fill {x1} {y1} {z1} {x2} {y2} {z2} minecraft:{material}"
        self.command(command)

    def check_blocks(self, blocks):
        for index in range(0, len(blocks), 3):
            group = blocks[index:index + 3]
            conditions = " ".join(
                f"if block {x} {y} {z} minecraft:{material}" for x, y, z, material in group
            )
            self.command(f"/execute {conditions} run give @p minecraft:paper 1")
            self.settle()
            if not any(slot["type"].split(":")[-1] == "paper"
                       for slot in self.info["inventory"].values()):
                raise RuntimeError(f"T1 /execute if block validation failed for {group}")
            self.command("/clear @p minecraft:paper")
            self.settle(2)

    def prepare_ground(self):
        self.command("/gamerule doMobSpawning false")
        self.command("/gamerule doDaylightCycle false")
        self.command("/gamerule doWeatherCycle false")
        self.command("/time set day")
        self.command("/weather clear")
        self.command("/difficulty peaceful")
        low_x, high_x = self.x - 110, self.x + 110
        low_z, high_z = self.z - 110, self.z + 110
        # One 37 x 221 x 4 strip fits the Minecraft 1.16.5 fill limit.
        for first_z in range(low_z, high_z + 1, 37):
            last_z = min(first_z + 36, high_z)
            self.fill(low_x, self.y - 1, first_z, high_x, self.y - 1, last_z, "grass_block")
            # Clear overhead terrain from the top down so falling blocks from
            # the remaining hills cannot enter the navigation surface.
            for elevation in (12, 8, 4, 0):
                self.fill(low_x, self.y + elevation, first_z,
                          high_x, self.y + elevation + 3, last_z, "air")
        samples = [(x, self.y - 1, z, "grass_block") for x in (low_x, self.x, high_x)
                   for z in (low_z, self.z, high_z)]
        samples += [(x, self.y + 1, z, "air") for x in (low_x, self.x, high_x)
                    for z in (low_z, self.z, high_z)]
        self.check_blocks(samples)

    def project(self, sideways, forward):
        dx, dz = self.direction
        return self.landmark[0] + dz * sideways + dx * forward, (
            self.landmark[1] - dx * sideways + dz * forward
        )

    def scene_box(self, first, last, material):
        x1, z1 = map(math.floor, first)
        x2, z2 = map(math.floor, last)
        box = min(x1, x2), self.y, min(z1, z2), max(x1, x2), self.y + 2, max(z1, z2)
        self.fill(*box, material)
        self.scene_fills.append(box)
        return box

    def prepare_episode(self, scenario_seed, distance):
        rng = random.Random(scenario_seed)
        self.direction = rng.choice(((0, 1), (-1, 0), (0, -1), (1, 0)))
        for box in self.scene_fills:
            self.fill(*box, "air")
        self.scene_fills.clear()
        gold = self.scene_box(self.landmark, self.landmark, "gold_block")
        side = rng.choice((-1, 1))
        obstacles = []
        for index, material in enumerate(("iron_block", "lapis_block", "redstone_block")):
            if index == 2 and rng.randrange(2) == 0:
                break
            location = self.project((side if index % 2 else -side) * (16 + 4 * index),
                                    rng.randint(2, distance - 2))
            obstacles.append(dict(material=material, horizontal=location))
            self.scene_box(location, location, material)
        wall = None
        halfway = distance // 2
        if distance >= 30:
            wall = self.scene_box(self.project(-3, halfway), self.project(3, halfway),
                                  "cobblestone")
        marker = [(gold[0], self.y, gold[2], "gold_block"),
                  (gold[0], self.y + 2, gold[2], "gold_block")]
        marker.extend((math.floor(item["horizontal"][0]), self.y + 2,
                       math.floor(item["horizontal"][1]), item["material"]) for item in obstacles)
        if wall:
            middle = self.project(0, halfway)
            marker.append((math.floor(middle[0]), self.y + 1,
                           math.floor(middle[1]), "cobblestone"))
        self.check_blocks(marker)

        endpoint = self.project(0, distance)
        spawn = self.project(side * 2, -2)
        waypoints = [self.project(side * rng.randint(7, 9), halfway)]
        if rng.randrange(2):
            waypoints.insert(0, self.project(side * rng.randint(5, 7), distance // 5))
        if rng.randrange(2):
            waypoints.append(self.project(side * rng.randint(5, 7), 4 * distance // 5))
        waypoints.append(endpoint)

        # Keep the gold pillar behind the agent for the final 20 steps.
        final_yaw = bearing(self.landmark, endpoint)
        initial_yaw = bearing(spawn, self.landmark)
        self.settle(8)
        self.command(f"/tp @p {spawn[0]} {self.y} {spawn[1]} {initial_yaw} 0")
        self.settle()
        if math.dist(horizontal(self.info), spawn) > 0.15:
            raise RuntimeError(f"Failed to restore T1 outbound start: {horizontal(self.info)}")
        self.history = deque(maxlen=2)
        self.outbound = []
        self.path_len_out = 0.0
        return dict(
            protocol="t1_v3", scenario_seed=scenario_seed, world_seed=self.world_seed,
            distance=distance, arena_origin=[self.x, self.y, self.z],
            native_spawn=self.native_spawn, spawn=self.info["player_pos"],
            landmark=self.landmark, distractors=obstacles,
            wall=wall, planned_endpoint=endpoint, waypoints=waypoints, final_yaw=final_yaw,
            outbound_dwell_steps=40, final_facing_steps=20,
        )

    def teacher_step(self, yaw_delta=0.0, forward=False, jump=False, record=False):
        action = self.sim.noop_action()
        action.update(camera=np.array([0.0, yaw_delta], dtype=np.float32),
                      forward=int(forward), jump=int(jump))
        if not record:
            self.tick(action)
            return
        batch = {key: np.asarray([value]) for key, value in action.items()}
        agent = self.sim.env_action_to_agent_action(batch)
        agent_action = (int(agent["buttons"].item()), int(agent["camera"].item()))
        action_token = self.tokenizer.encode_action(agent_action)
        previous = horizontal(self.info)
        self.history.append((self.info["pov"].copy(), action_token))
        self.tick(action)
        current = horizontal(self.info)
        self.path_len_out += math.dist(previous, current)
        self.outbound.append([*current, float(self.info["player_pos"]["yaw"]), action_token])

    def face(self, yaw, record=False):
        for step in range(30):
            error = wrapped_angle(yaw - self.info["player_pos"]["yaw"])
            if abs(error) <= 2:
                return step
            self.teacher_step(yaw_delta=max(-10, min(10, error)), record=record)
        raise RuntimeError("Teacher could not reach required heading")

    def drive(self, target, tolerance, budget, record=False, on_step=None):
        closest = math.dist(horizontal(self.info), target)
        stalled = 0
        for step in range(budget + 1):
            position = horizontal(self.info)
            distance = math.dist(position, target)
            if distance <= tolerance or step == budget:
                return step, distance <= tolerance, min(closest, distance)
            error = wrapped_angle(bearing(position, target) - self.info["player_pos"]["yaw"])
            forward = abs(error) <= 8
            self.teacher_step(max(-10, min(10, error)), forward,
                              forward and stalled >= 5 and stalled % 6 == 5, record)
            if on_step:
                on_step()
            advanced = math.dist(position, horizontal(self.info)) > 0.02
            stalled = 0 if advanced or not forward else stalled + 1
            closest = min(closest, distance)
        raise AssertionError("unreachable teacher state")

    def outbound_segment(self, config):
        for _ in range(40):
            self.teacher_step(record=True)
        for target in config["waypoints"]:
            _, reached, _ = self.drive(target, POINT_TOLERANCE, 800, record=True)
            if not reached:
                raise RuntimeError(f"Teacher could not reach outbound waypoint {target}")
        self.face(config["final_yaw"], record=True)
        for _ in range(20):
            self.teacher_step(record=True)
        config["return_pose"] = self.info["player_pos"]
        config["actual_distance"] = math.dist(horizontal(self.info), self.landmark)
        config["path_len_out"] = self.path_len_out
        if abs(config["actual_distance"] - config["distance"]) > POINT_TOLERANCE + 0.2:
            raise RuntimeError(f"Outbound distance out of tolerance: {config['actual_distance']}")
        if any(abs(wrapped_angle(row[2] - config["final_yaw"])) > 2
               for row in self.outbound[-20:]):
            raise RuntimeError("Landmark heading is invalid at start of return")

    def teacher_return(self, config):
        distance = config["distance"]
        routes = [[self.landmark]]
        if config["wall"]:
            halfway = distance // 2
            routes = [[self.project(edge, halfway + 1.5), self.project(edge, halfway - 1.5),
                       self.landmark] for edge in (-4.5, 4.5)]
        candidates = []
        for index, targets in enumerate(routes):
            if index:
                self.restore_return_pose(config)
            candidates.append((*self.follow_return(targets, distance), targets))
        success, steps, final_dist, min_dist, targets = min(
            candidates, key=lambda candidate: (not candidate[0], candidate[1], candidate[2])
        )
        config["teacher_targets"] = targets
        config["teacher_route_attempts"] = [
            dict(success=result[0], steps=result[1], final_dist=result[2], targets=result[4])
            for result in candidates
        ]
        config["teacher_return_steps"] = steps
        if success:
            config["return_limit"] = 2 * steps + 200
        return success, steps, final_dist, min_dist

    def restore_return_pose(self, config):
        self.sim.action_type = "env"
        pose = config["return_pose"]
        self.settle(8)
        # Minecraft's command parser rejects angles written in scientific notation.
        coords = " ".join(f"{pose[key]:.6f}" for key in ("x", "y", "z", "yaw", "pitch"))
        self.command(f"/tp @p {coords}")
        self.settle(6)
        if math.dist(horizontal(self.info), (pose["x"], pose["z"])) > 0.15:
            raise RuntimeError(f"Failed to restore T1 return pose: got {horizontal(self.info)}, "
                               f"expected {(pose['x'], pose['z'])}")

    def follow_return(self, targets, distance):
        steps = 0
        min_dist = math.dist(horizontal(self.info), self.landmark)
        success = False

        def measure_progress():
            nonlocal min_dist
            min_dist = min(min_dist, math.dist(horizontal(self.info), self.landmark))

        for index, target in enumerate(targets):
            final = index == len(targets) - 1
            count, reached, _ = self.drive(target, SUCCESS_RADIUS if final else POINT_TOLERANCE,
                                           8 * distance + 200 - steps, on_step=measure_progress)
            steps += count
            if not reached:
                break
            success = final
        return success, steps, math.dist(horizontal(self.info), self.landmark), min_dist


def teacher_worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "teacher.csv").open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("distance", "seed", "world_seed", "success", "steps", "final_dist",
                         "min_dist", "path_len_out", "actual_distance", "error"))
        for world in range(args.worker, min(WORLD_COUNT, args.seeds), args.workers):
            arena = Arena(WORLD_REPLACEMENTS.get(world, WORLD_SEED + world))
            try:
                for index in range(world, args.seeds, WORLD_COUNT):
                    for distance in DISTANCES:
                        seed = TEACHER_SEED + index
                        metrics = teacher_episode(args.output, arena, seed, distance)
                        writer.writerow((distance, seed, arena.world_seed, *metrics))
                        output.flush()
                        print(distance, seed, metrics, flush=True)
            finally:
                arena.sim.close()


def teacher_episode(output, arena, seed, distance):
    config = None
    name = f"d{distance}-seed{seed}"
    try:
        config = arena.prepare_episode(seed, distance)
        arena.outbound_segment(config)
        success, steps, final_dist, min_dist = arena.teacher_return(config)
        metrics = (success, steps, final_dist, min_dist, config["path_len_out"],
                   config["actual_distance"], "")
    except RuntimeError as error:
        metrics = (False, "", "", "", "", "", str(error))
    if config is not None:
        (output / f"{name}.json").write_text(json.dumps(config, indent=2, default=float) + "\n")
        with (output / f"{name}-outbound.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "z", "yaw", "action_token"))
            writer.writerows(arena.outbound)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers <= min(WORLD_COUNT, args.seeds):
        parser.error("Use positive seed/worker counts and a worker index in range")
    teacher_worker(args)


if __name__ == "__main__":
    main()

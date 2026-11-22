"""Instrument T1 trials and relocate respawn outside the frozen arena."""

import csv
import gzip
import math
from contextlib import contextmanager

from t1_v4 import RandomHeadingArena, horizontal


SETUP_REVISION = "remote_respawn_v1"
PREVENTION_COMMANDS = (
    "/gamerule doMobSpawning false",
    "/kill @e[type=!player]",
    "/time set day",
    "/gamerule doDaylightCycle false",
    "/gamerule doWeatherCycle false",
    "/weather clear",
)
EFFECT_COMMANDS = (
    "/effect give @p resistance 99999 4 true",
    "/effect give @p water_breathing 99999 0 true",
)
POSE_FIELDS = ("x", "y", "z", "yaw", "pitch")
STEP_FIELDS = ("step", *POSE_FIELDS, "health", "is_alive", "death_count",
               "death", "pose_jump", "terminated", "truncated")


def life_values(info):
    life = info.get("life_stats") or {}
    health = info.get("health", life.get("life"))
    alive = life.get("is_alive")
    custom = info.get("custom", {})
    counter = next((custom[key] for key in ("deaths", "minecraft:deaths") if key in custom), 0)
    return (float(health) if health is not None else None,
            bool(alive) if alive is not None else None, int(counter))


class StepTrace:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.deaths = 0
        self.pose_jumps = 0
        self.reason = ""
        self._counter_deaths = 0
        self._health_deaths = 0
        self._dead = False
        self._previous_counter = 0

    def observe(self, info, terminated=False, truncated=False):
        pose = tuple(float(info["player_pos"][field]) for field in POSE_FIELDS)
        health, alive, counter = life_values(info)
        dead = (health is not None and health <= 0) or alive is False
        previous = self.rows[-1] if self.rows else None
        if previous:
            self._counter_deaths += max(0, counter - self._previous_counter)
            self._health_deaths += int(dead and not self._dead)
        else:
            self._health_deaths += int(dead)
        new_deaths = max(self._counter_deaths, self._health_deaths)
        death = new_deaths > self.deaths
        self.deaths = new_deaths
        self._previous_counter, self._dead = counter, dead
        pose_jump = not all(map(math.isfinite, pose[:3]))
        if previous:
            pose_jump |= math.dist((previous[1], previous[3]), (pose[0], pose[2])) > 2
        self.pose_jumps += int(pose_jump)
        self.rows.append((len(self.rows), *pose, health, alive, counter, death, pose_jump,
                          bool(terminated), bool(truncated)))
        if death or pose_jump or terminated or truncated:
            self.reason = "death" if death else ("pose_jump" if pose_jump else "terminated")
            return False
        return True

    def flags(self):
        return self.deaths, bool(self.pose_jumps), self.pose_jumps

    def write(self):
        if self.path is None:
            return
        with gzip.open(self.path, "xt", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(STEP_FIELDS)
            writer.writerows(self.rows)


class TrialInterrupted(RuntimeError):
    pass


class ProtectedArena(RandomHeadingArena):
    def __init__(self, world_seed, effects=False):
        self.trace = None
        self.effects = effects
        self.teacher_traces = None
        super().__init__(world_seed)

    def run_commands(self, commands):
        from minestudio.simulator.callbacks.commands import CommandsCallback

        callback = CommandsCallback(list(commands))
        _, self.info = callback.after_reset(self.sim, {"pov": self.info["pov"]}, self.info)

    def prepare_ground(self):
        super().prepare_ground()
        x, y, z = self.x + 384, max(self.y, 160), self.z + 384
        self.respawn_point = (x, y, z)
        load = f"/forceload add {x-1} {z-1} {x+1} {z+1}"
        self.respawn_pad_commands = (
            load,
            f"/fill {x-1} {y-1} {z-1} {x+1} {y-1} {z+1} minecraft:stone",
            f"/fill {x-1} {y} {z-1} {x+1} {y+2} {z+1} minecraft:air",
        )
        self.run_commands([load])
        self.settle(12)
        self.run_commands(self.respawn_pad_commands[1:])
        self.protection_commands = (*PREVENTION_COMMANDS, f"/spawnpoint @p {x} {y} {z}",
                                    *(EFFECT_COMMANDS if self.effects else ()))
        self.run_commands(self.protection_commands)
        self.settle()

    def safety_config(self):
        return dict(setup_revision=SETUP_REVISION, protection_commands=self.protection_commands,
                    respawn_pad_commands=self.respawn_pad_commands, respawn_point=self.respawn_point,
                    optional_effects=self.effects)

    def prepare_episode(self, scenario_seed, distance):
        self.run_commands(self.protection_commands)
        config = super().prepare_episode(scenario_seed, distance)
        if any(math.dist((self.respawn_point[0], self.respawn_point[2]), point) < 150
               for point in [config["landmark"], *config["waypoints"]]):
            raise RuntimeError("Respawn point is too close to the T1 scene")
        config.update(self.safety_config())
        return config

    @contextmanager
    def track(self, trace):
        if self.trace is not None:
            raise RuntimeError("Cannot nest T1 control traces")
        self.trace = trace
        try:
            if not trace.observe(self.info):
                raise TrialInterrupted(trace.reason)
            yield
        finally:
            self.trace = None
            trace.write()

    def navigation_step(self, action):
        if self.trace is None:
            raise RuntimeError("Protected T1 control requires a step trace")
        _, _, ended, truncated, self.info = self.sim.step(action)
        return self.trace.observe(self.info, ended, truncated)

    def tick(self, action=None):
        if self.trace is None:
            super().tick(action)
        elif not self.navigation_step(self.sim.noop_action() if action is None else action):
            raise TrialInterrupted(self.trace.reason)

    def follow_return(self, targets, distance):
        if self.teacher_traces is None:
            return super().follow_return(targets, distance)
        folder, name, trials = self.teacher_traces
        trace = StepTrace(folder / f"{name}-teacher-route-{len(trials)+1}.csv.gz")
        trials.append(trace)
        with self.track(trace):
            return super().follow_return(targets, distance)


def total_flags(traces):
    deaths = sum(trace.deaths for trace in traces)
    pose_jumps = sum(trace.pose_jumps for trace in traces)
    return deaths, bool(pose_jumps), pose_jumps

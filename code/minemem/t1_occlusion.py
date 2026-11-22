"""Candidate T1 walls, screened landmark and isolated Minecraft settings."""

import math
import random
from pathlib import Path

from t1_safety import StepTrace


SCREENED_PROTOCOL = "t1_occlusion_candidate"


class OcclusionGeometry:
    """Four visually matching walls around each distant return endpoint."""

    screened_pillar = False

    def turn_to(self, yaw, record=False):
        # The live camera can briefly return to its commanded starting yaw.
        # Count those steps in the same control budget while the turn converges.
        return super().turn_to(yaw, record=record, max_steps=48)

    def tall_wall(self, s1,t1,s2,t2):
        box = self.scene_box(self.project(s1,t1),self.project(s2,t2),"cobblestone")
        box = (*box[:4],self.y+5,box[5])
        self.scene_fills[-1] = box
        self.fill(*box,"cobblestone")
        return box

    def prepare_episode(self, seed, distance):
        config = super().prepare_episode(seed, distance)
        config.update(protocol=SCREENED_PROTOCOL, render_distance_chunks=4,
                      occlusion_walls=[], screened_pillar=self.screened_pillar)
        old = config["wall"]
        if old is None:
            return config
        self.fill(*old, "air")
        self.scene_fills.remove(old)
        rng = random.Random(f"occlusion:{seed}:{distance}")
        near, far, left, right = [rng.randint(9,10) for _ in range(4)]
        centers = ((0,distance-near), (0,distance+far), (-left,distance), (right,distance))
        wall_specs = []
        for i, (s,t) in enumerate(centers):
            panel = (s-6,t,s+6,t) if i<2 else (s,t-6,s,t+6)
            wall_specs.append(self.tall_wall(*panel))
        check = [(math.floor(x), self.y+5, math.floor(z)) for x,z in
                 (self.project(s,t) for s,t in centers)]
        self.check_blocks([(*point,"cobblestone") for point in check])
        real = distance-near
        side = rng.choice((-1,1))
        config.update(wall=wall_specs[0], occlusion_walls=wall_specs,
                      waypoints=[self.project(side*7.2,real-2), self.project(side*7.2,real+2),
                                 config["planned_endpoint"]])
        config["candidate_routes"] = [
            [self.project(edge,real+2), self.project(edge,real-2), self.landmark]
            for edge in (-7.2,7.2)]
        if self.screened_pillar:
            spawn = config["spawn"]
            sx,sz = self.project(1,0)
            lx,lz = self.landmark
            sign = 1 if (spawn["x"]-lx)*(sx-lx)+(spawn["z"]-lz)*(sz-lz)>0 else -1
            door, edge = 2*sign, 7.2*sign
            panels = ((-5,-5,-5,5),(5,-5,5,5),(-5,5,5,5),
                      (-5,-5,door-2,-5),(door+2,-5,5,-5),(-5,-9,5,-9))
            config["pillar_screen"] = [self.tall_wall(*panel) for panel in panels]
            config["waypoints"] = [self.project(door,-7),self.project(edge,-7),
                                   self.project(edge,real+2),config["planned_endpoint"]]
            config["candidate_routes"] = [[self.project(edge,real+2),self.project(edge,-7),
                                           self.project(door,-7),self.project(door,-3),self.landmark]]
        return config

    def teacher_return(self, config):
        if not config["wall"]:
            return super().teacher_return(config)
        scores = []
        for path in config["candidate_routes"]:
            self.restore_return_pose(config)
            trace = StepTrace(None)
            with self.track(trace):
                scores.append((*self.follow_return(path, config["distance"]), path))
        success, steps, final, closest, targets = min(scores, key=lambda row:(not row[0],row[1],row[2]))
        config.update(teacher_targets=targets, teacher_return_steps=steps, return_limit=2*steps+200,
                      teacher_route_attempts=[dict(success=trial[0], steps=trial[1], final_dist=trial[2],
                                                   targets=trial[4]) for trial in scores])
        return success,steps,final,closest


def set_candidate_render_distance(output):
    from minestudio.utils import get_mine_studio_dir
    import minestudio

    template = Path(minestudio.__file__).parent/"simulator/minerl/env/runtime/options.txt"
    instance = Path(get_mine_studio_dir()).resolve()
    if instance != (output/"minestudio").resolve():
        raise ValueError("Candidate render settings require an isolated simulator directory")
    runtime = instance/"runtime"
    runtime.mkdir(parents=True,exist_ok=True)
    options = runtime/"options.txt"
    base = options.read_text() if options.exists() else template.read_text()
    entries = [line for line in base.splitlines() if not line.startswith("renderDistance:")]
    options.write_text("\n".join([*entries,"renderDistance:4"])+"\n")

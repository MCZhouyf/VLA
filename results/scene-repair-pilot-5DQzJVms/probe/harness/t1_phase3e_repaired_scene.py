"""Prepare the revised dual-enclosure scenery using the frozen T1 v3 mechanics."""

import random

from t1_dual_enclosure import candidate_occlusion
from t1_phase3e import FirstArrivalArena
from t1_phase3e_pairs import PairedArena
from t1_v4 import bearing, horizontal


SCENE_REVISION = "staggered_symmetric_screens"


class RepairedScene(FirstArrivalArena):
    def prepare_ground(self):
        bounds=f"{self.x-110} {self.z-110} {self.x+110} {self.z+110}"
        self.command(f"/forceload add {bounds}")
        self.settle(32)
        try:
            FirstArrivalArena.prepare_ground(self)
        finally:
            self.command(f"/forceload remove {bounds}")
        self.settle(6)

    def layout(self, seed, distance):
        layout=super().layout(seed,distance)
        gates=[]
        if distance>=30:
            # The openings depend on the scenario seed, independently of the planted material.
            first=random.Random(f"{SCENE_REVISION}:{seed}").choice((-5,5))
            gates=[(first,8),(-first,17)]
        layout.update(scene_revision=SCENE_REVISION,neutral_gates=gates,ground_force_loaded=True)
        return layout

    def establish(self, layout, goal_slot):
        super().establish(layout,goal_slot)
        samples=[]
        for opening,depth in layout["neutral_gates"]:
            for lo,hi in ((-28,opening-3),(opening+3,28)):
                box=self.panel((lo,depth),(hi,depth),0,6,"cobblestone")
                layout["occlusion_boxes"].append(box)
                x,_,z,_,top,_=box
                samples.append((x,top,z,"cobblestone"))
        if samples:
            self.check_blocks(samples)
        layout["walls"]=self.scene_fills.copy()
        layout["candidate_occlusion"]=candidate_occlusion(layout,self.y)
        if not all(layout["candidate_occlusion"]):
            raise RuntimeError("A candidate is exposed through the neutral screens")

    def gate_route(self, layout):
        return [self.project(side,depth+offset)
                for side,depth in layout["neutral_gates"] for offset in (-3,3)]

    def traverse_outbound(self, layout):
        gates=self.gate_route(layout)
        if not gates:
            return super().traverse_outbound(layout)
        prefix=dict(layout,endpoint=layout["corridor"],return_pose=dict(
            layout["return_pose"],yaw=bearing(layout["corridor"],gates[0])))
        cues=super().traverse_outbound(prefix)
        layout["room_frame_indices"]=prefix["room_frame_indices"]
        for point in (*gates,layout["endpoint"]):
            _,reached,_=self.drive(point,.35,850,record=True)
            if not reached:
                raise RuntimeError(f"Outbound teacher missed neutral-screen route point {point}")
        self.face(layout["return_pose"]["yaw"],record=True)
        return cues

    def return_targets(self, layout, slot):
        return [*reversed(self.gate_route(layout)),*super().return_targets(layout,slot)]


class RepairedPairedScene(RepairedScene,PairedArena):
    """Render swapped materials on the identical revised layout."""


class GroundProbe(FirstArrivalArena):
    def check_blocks(self, blocks):
        try:
            return super().check_blocks(blocks)
        except RuntimeError as error:
            print(f"Original ground check: {error}",flush=True)
            bounds=f"{self.x-110} {self.z-110} {self.x+110} {self.z+110}"
            self.command(f"/forceload add {bounds}")
            self.settle(32)
            missing=[]
            for block in blocks:
                try:
                    PairedArena.check_materials(self,[block])
                except RuntimeError:
                    missing.append(block)
            print(f"After loading, without refilling, failing blocks: {missing}",flush=True)
            raise

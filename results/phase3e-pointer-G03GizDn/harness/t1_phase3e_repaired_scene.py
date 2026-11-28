"""Prepare the revised dual-enclosure scenery using the frozen T1 v3 mechanics."""

import random

from t1_dual_enclosure import candidate_occlusion
from t1_phase3e import FirstArrivalArena
from t1_phase3e_pairs import PairedArena
from t1_v4 import bearing, horizontal


SCENE_REVISION = "staggered_screens_with_blind_approach"
OPENING_NAMESPACE = "staggered_symmetric_screens"


def matches_block(arena, block):
    try:
        PairedArena.check_materials(arena,[block])
    except RuntimeError:
        return False
    return True


class GroundRepairedArena(FirstArrivalArena):
    """Use the original v3 layout with local clearance of failed air samples."""

    def prepare_ground(self):
        self.ground_repairs=[]
        self.checking_ground=True
        bounds=f"{self.x-110} {self.z-110} {self.x+110} {self.z+110}"
        self.command(f"/forceload add {bounds}")
        self.settle(32)
        try:
            FirstArrivalArena.prepare_ground(self)
        finally:
            self.checking_ground=False
            self.command(f"/forceload remove {bounds}")
        self.settle(6)

    def check_blocks(self, blocks):
        try:
            return super().check_blocks(blocks)
        except RuntimeError:
            if not self.checking_ground:
                raise
            failures=[block for block in blocks if not matches_block(self,block)]
            if any(block[3]!="air" for block in failures):
                raise
            for x,y,z,_ in failures:
                # Clear the whole column near a failing sample without altering the platform.
                bounds=(max(self.x-110,x-2),self.y,max(self.z-110,z-2),
                        min(self.x+110,x+2),255,min(self.z+110,z+2))
                self.fill(*bounds,"air")
                self.ground_repairs.append(dict(sample=[x,y,z], cleared_bounds=bounds))
            self.settle(32)
            PairedArena.check_materials(self,blocks)
            print(f"Verified ground after local column clearance: {self.ground_repairs}",flush=True)

    def layout(self, seed, distance):
        layout=super().layout(seed,distance)
        layout.update(ground_force_loaded=True,ground_repairs=self.ground_repairs)
        return layout


class RepairedScene(GroundRepairedArena):
    def layout(self, seed, distance):
        layout=super().layout(seed,distance)
        gates=[]
        if distance>=30:
            # The openings depend on the scenario seed, independently of the planted material.
            first=random.Random(f"{OPENING_NAMESPACE}:{seed}").choice((-5,5))
            gates=[(first,8),(-first,17)]
        blind_screen=(gates[-1][0],22) if gates else None
        layout.update(scene_revision=SCENE_REVISION,neutral_gates=gates,blind_screen=blind_screen)
        return layout

    def establish(self, layout, goal_slot):
        super().establish(layout,goal_slot)
        panels=[]
        for opening,depth in layout["neutral_gates"]:
            panels.extend([(lo,hi,depth) for lo,hi in ((-28,opening-3),(opening+3,28))])
        if layout["blind_screen"]:
            center,depth=layout["blind_screen"]
            panels.append((center-4,center+4,depth))
        samples=[]
        for lo,hi,depth in panels:
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
        route=[self.project(side,depth+offset)
               for side,depth in layout["neutral_gates"] for offset in (-3,3)]
        if layout["blind_screen"]:
            _,depth=layout["blind_screen"]
            route.extend((self.project(0,depth-2),self.project(0,depth+2)))
        return route

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
            missing=[block for block in blocks if not matches_block(self,block)]
            print(f"After loading, without refilling, failing blocks: {missing}",flush=True)
            kinds=("stone","dirt","grass_block","gravel","sand","water","tall_grass","grass",
                   "oak_leaves","birch_leaves","vine","snow","air")
            for x,y,z,_ in missing:
                for level in (y,y+4,y+12,y+16):
                    material=next((m for m in kinds if matches_block(self,(x,level,z,m))),"unidentified")
                    print(f"Observed block {x},{level},{z}: {material}",flush=True)
            raise

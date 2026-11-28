"""Learn goal and lookahead pointers over the visited cells of an outbound route."""

import math

import numpy as np
import torch
from torch import nn

from t1_phase3e_grid_plan import heading_turn
from t1_waypoint import waypoint_instruction, waypoint_labels


def nearest_cells(positions, targets):
    targets = torch.as_tensor(targets, dtype=positions.dtype, device=positions.device)
    return (positions[None]-targets.reshape(-1, 1, 2)).square().sum(-1).argmin(-1)


class CellMemory:
    """One memory entry per visited block, with uniformly sampled frame features."""

    def __init__(self, outbound, sampled_poses, appearance):
        poses = np.asarray(outbound, dtype=np.float64)
        samples = np.asarray(sampled_poses, dtype=np.float64)
        if len(poses)<2 or len(samples)!=len(appearance) or not len(samples):
            raise ValueError("Cell memory needs an outbound route and aligned visual samples")
        self.origin = poses[0,:2]
        axis = poses[-1,:2] - self.origin
        self.axis = axis / np.linalg.norm(axis) if np.any(axis) else np.array((0.,1.))
        self.rotation = np.array(((self.axis[1], self.axis[0]), (-self.axis[0], self.axis[1])))
        occupied = np.floor(poses[:,:2]).astype(np.int64)
        groups = {}
        for index, coords in enumerate(occupied):
            groups.setdefault(tuple(coords), []).append(index)
        sampled = {}
        for index, coords in enumerate(np.floor(samples[:,:2]).astype(np.int64)):
            sampled.setdefault(tuple(coords), []).append(index)
        points, entries, times = [], [], []
        for coords, indices in groups.items():
            visited = poses[indices,:2]
            index = ((visited-(np.asarray(coords)+.5))**2).sum(1).argmin()
            point = visited[index]
            points.append(point)
            views = sampled.get(coords)
            missing = views is None
            if missing:
                views = [int(((samples[:,:2]-point)**2).sum(1).argmin())]
            visual = appearance[views]
            entries.append(torch.cat((visual.mean(0),visual.amax(0))))
            times.append((indices[0]/len(poses),indices[-1]/len(poses),float(missing)))
        self.world_points = [tuple(point) for point in points]
        self.points = appearance.new_tensor(np.asarray(points))
        self.local_points = self.local_pose(np.column_stack((points,np.zeros(len(points)))))[:,:2]
        self.features = torch.cat((torch.stack(entries),self.local_points/24,
                                   appearance.new_tensor(times)),dim=-1)

    def local_pose(self, poses):
        poses = np.atleast_2d(np.asarray(poses,dtype=np.float64))
        point = (poses[:,:2]-self.origin)@self.rotation
        angle = np.deg2rad(poses[:,2])
        heading = np.stack((-np.sin(angle),np.cos(angle)),axis=-1) @ self.rotation
        return self.points.new_tensor(np.column_stack((point,heading)))


class CellPointers(nn.Module):
    def __init__(self, visual_width, width=64):
        super().__init__()
        self.entries = nn.Sequential(nn.Linear(visual_width*2+5,width),nn.LayerNorm(width),nn.GELU())
        layer = nn.TransformerEncoderLayer(width,4,128,dropout=.1,batch_first=True,norm_first=False)
        self.context = nn.TransformerEncoder(layer,1)
        self.goal = nn.Linear(width,1)
        self.waypoint_key = nn.Linear(width,width,bias=False)
        self.waypoint_query = nn.Sequential(nn.Linear(width+6,width),nn.LayerNorm(width),nn.GELU())
        self.relation = nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,1))

    def encode(self, memory):
        entries = self.context(self.entries(memory.features)[None])[0]
        return entries,self.goal(entries).flatten()

    def waypoints(self, memory, entries, local, goal):
        target = memory.local_points[goal]
        goal_delta = (target-local[:,:2])/12
        goal_entries = entries[goal].expand(len(local),-1)
        queries = self.waypoint_query(torch.cat((local[:,:2]/24,local[:,2:],goal_delta,goal_entries),-1))
        pointer = queries @ self.waypoint_key(entries).T / math.sqrt(entries.shape[-1])
        ahead = local[:,None,2:]
        sideways = torch.stack((ahead[...,1],-ahead[...,0]),dim=-1)
        offset = memory.local_points[None]-local[:,None,:2]
        to_goal = (memory.local_points-target)[None].expand(len(local),-1,-1)
        relative = torch.cat((offset/6,to_goal/12,(offset*ahead).sum(-1,keepdim=True)/6,
                              (offset*sideways).sum(-1,keepdim=True)/6),dim=-1)
        return pointer+self.relation(relative).squeeze(-1)


class PointerPolicy:
    """Commit to a learned world location for up to 20 controller steps."""

    def __init__(self, model, memory):
        self.model,self.memory = model,memory
        self.target = None
        self.remaining = 0
        self.turn = None
        with torch.inference_mode():
            self.entries,scores = model.encode(memory)
            self.goal = int(scores.argmax())
            self.goal_point = memory.world_points[self.goal]

    @torch.inference_mode()
    def __call__(self, info):
        pose = tuple(float(info["player_pos"][key]) for key in ("x","z","yaw"))
        if self.target is None or self.remaining==0 or math.dist(pose[:2],self.target)<=1.5:
            scores = self.model.waypoints(self.memory,self.entries,self.memory.local_pose(pose),self.goal)
            self.target = self.memory.world_points[int(scores.argmax())]
            self.remaining = 20
        self.remaining -= 1
        self.turn = heading_turn(pose,self.target)
        return waypoint_instruction(*waypoint_labels(pose,self.target))

    def corrective_yaw(self):
        return self.turn

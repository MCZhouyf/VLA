"""Fit and replay an action-only flat-ground dead-reckoning baseline."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


def motion_direction(yaw, buttons):
    forward = buttons.get("forward",0)-buttons.get("back",0)
    lateral = buttons.get("right",0)-buttons.get("left",0)
    normal = max(1.,math.hypot(forward,lateral))
    angle = math.radians(yaw)
    return np.array((-forward*math.sin(angle)-lateral*math.cos(angle),
                     forward*math.cos(angle)-lateral*math.sin(angle)))/normal


class ActionIntegrator:
    def __init__(self, drag, acceleration):
        self.drag,self.acceleration = drag,acceleration
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.yaw = 0.

    def step(self, camera_yaw, buttons):
        self.yaw = (self.yaw+camera_yaw+180)%360-180
        self.velocity = self.drag*self.velocity+self.acceleration*motion_direction(self.yaw,buttons)
        self.position = self.position+self.velocity
        return float(self.position[0]),float(self.position[1]),self.yaw


def traces(source):
    for log in sorted(source.rglob("*-actions.jsonl")):
        path = log.with_name(log.name.removesuffix("-actions.jsonl")+"-return.csv.gz")
        if not path.is_file():
            continue
        with gzip.open(path,"rt",newline="") as stream:
            rows = list(csv.DictReader(stream))
        actions = list(map(json.loads,log.read_text().splitlines()))
        if len(rows)!=len(actions)+1 or not actions:
            raise ValueError(f"Misaligned commands and return poses: {log}")
        if any(row["death"]=="True" or row["pose_jump"]=="True" for row in rows):
            continue
        pose = np.array([[float(r[key]) for key in ("x","z","yaw")] for r in rows])
        seed = int(log.name.split("-seed")[1].split("-")[0])
        yield seed, pose, actions


def fit(cases):
    design, targets = [], []
    for _,actual,actions in cases:
        displacement = np.diff(actual[:,:2],axis=0)
        previous = np.vstack((np.zeros(2),displacement[:-1]))
        directions = np.stack([motion_direction(pose[2],action["buttons"])
                               for pose,action in zip(actual[1:],actions,strict=True)])
        design.append(np.column_stack((previous.ravel(),directions.ravel())))
        targets.extend(displacement.ravel())
    drag, acceleration = np.linalg.lstsq(np.concatenate(design),targets,rcond=None)[0]
    if not (0<=drag<1 and acceleration>0):
        raise ValueError("The fitted ground dynamics are unstable")
    return float(drag),float(acceleration)


def audit(cases, drag, acceleration):
    rows = []
    for seed,actual,actions in cases:
        integration = ActionIntegrator(drag,acceleration)
        predicted = np.array([integration.step(action["yaw"],action["buttons"]) for action in actions])
        # Global alignment is used only to score the local action-integrated trajectory.
        angle = math.radians(actual[0,2])
        rotation = np.array(((math.cos(angle),math.sin(angle)),(-math.sin(angle),math.cos(angle))))
        aligned = actual[0,:2]+predicted[:,:2]@rotation
        error = np.linalg.norm(aligned-actual[1:,:2],axis=1)
        heading = (predicted[:,2]+actual[0,2]-actual[1:,2]+180)%360-180
        rows.append(dict(seed=seed,steps=len(error),horizontal_rmse=math.sqrt(np.mean(error**2)),
                         final_error=float(error[-1]),yaw_rmse=math.sqrt(np.mean(heading**2))))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    cases = list(traces(args.source))
    if len({seed for seed,_,_ in cases})!=len(cases):
        raise ValueError("Calibration expects a single trace per scenario")
    train = [case for case in cases if case[0]%2==0]
    test = [case for case in cases if case[0]%2!=0]
    drag,acceleration = fit(train)
    rows = audit(test,drag,acceleration)
    result = dict(source=str(args.source),fit_seeds=[case[0] for case in train],test_seeds=[r["seed"] for r in rows],
                  drag=drag,acceleration=acceleration,estimated_steady_speed=acceleration/(1-drag),
                  median_final_error=median(r["final_error"] for r in rows),
                  median_horizontal_rmse=median(r["horizontal_rmse"] for r in rows),
                  median_yaw_rmse=median(r["yaw_rmse"] for r in rows),cases=rows)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"}),flush=True)

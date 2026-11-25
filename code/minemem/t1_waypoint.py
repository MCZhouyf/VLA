"""One waypoint representation shared by teacher, VLA controller, and memory heads."""

import math
import re


INSTRUCTION = "Follow the waypoint."
LOOKAHEAD_BLOCKS = 6.0
DIRECTION_BINS = 36
NEAR_WAYPOINT = re.compile(r"waypoint: (ahead 0|left 10|right 10) deg, ([1-6]) blocks")
NEAR_ANGLES = {"ahead 0": 0, "left 10": -10, "right 10": 10}
STATIONARY_TURN_BUTTONS = ("forward", "back", "left", "right", "jump", "attack", "use")
STABILIZER_ALTERNATING_TURNS = 24
STABILIZER_FORWARD_STEPS = 4


def relative_waypoint(pose, target):
    x, z, yaw = pose
    bearing = math.degrees(math.atan2(x-target[0], target[1]-z))
    return ((bearing-yaw+180) % 360)-180, math.dist((x, z), target)


def waypoint_labels(pose, target):
    angle, distance = relative_waypoint(pose, target)
    return round(angle/10) % DIRECTION_BINS, round(distance)


def waypoint_instruction(direction, distance):
    angle = (direction % DIRECTION_BINS) * 10
    if angle > 180:
        angle -= 360
    side = "right" if angle > 0 else "left" if angle < 0 else "ahead"
    return f"{INSTRUCTION}\nwaypoint: {side} {abs(angle)} deg, {max(0, round(distance))} blocks"


class WaypointTurnStabilizer:
    """Advance briefly when the policy repeatedly turns away from an aligned waypoint."""

    def __init__(self):
        self.previous = None
        self.turns = 0
        self.forward_steps = 0

    def should_advance(self, prompt, action):
        match = NEAR_WAYPOINT.fullmatch(prompt.rsplit("\n", 1)[-1])
        angle = NEAR_ANGLES[match[1]] if match else None
        if self.forward_steps and angle == 0:
            self.forward_steps -= 1
            return True
        self.forward_steps = 0

        pitch, yaw = map(float, action["camera"])
        stationary_turn = (match is not None and abs(yaw) == 10 and pitch == 0
                           and not any(action.get(button) for button in STATIONARY_TURN_BUTTONS)
                           and (angle == 0 or angle == yaw))
        if not stationary_turn:
            self.previous = None
            self.turns = 0
            return False

        distance = match[2]
        previous = self.previous
        alternates = previous is not None and previous == (angle - yaw, distance, -yaw)
        self.turns = self.turns + 1 if alternates else 1
        self.previous = (angle, distance, yaw)
        if self.turns < STABILIZER_ALTERNATING_TURNS or angle != 0:
            return False
        self.previous = None
        self.turns = 0
        self.forward_steps = STABILIZER_FORWARD_STEPS - 1
        return True


def project(point, start, end):
    dx, dz = end[0]-start[0], end[1]-start[1]
    squared = dx*dx+dz*dz
    position = ((point[0]-start[0])*dx+(point[1]-start[1])*dz)/squared if squared else 0.0
    fraction = max(0.0, min(1.0, position))
    return fraction, (start[0]+fraction*dx, start[1]+fraction*dz)


class RouteLookahead:
    def __init__(self, start, targets):
        self.points = (tuple(start), *(tuple(point) for point in targets))
        if len(self.points) < 2:
            raise ValueError("A waypoint route needs a start and at least one target")
        self.segment = 0

    def target(self, pose):
        point = pose[:2]
        last = len(self.points)-2
        while self.segment < last:
            endpoint = self.points[self.segment+1]
            fraction, current_projection = project(point, self.points[self.segment], endpoint)
            along, following = project(point, endpoint, self.points[self.segment+2])
            passed = (along > 0 and math.dist(point, following)+.2 < math.dist(point, current_projection)
                      and math.dist(point, endpoint) <= LOOKAHEAD_BLOCKS)
            if math.dist(point, endpoint) > .45 and not (passed or fraction == 1.0):
                break
            self.segment += 1
        _, following = project(point, self.points[self.segment], self.points[self.segment+1])
        remaining = LOOKAHEAD_BLOCKS
        for endpoint in self.points[self.segment+1:]:
            length = math.dist(following, endpoint)
            if length > remaining:
                fraction = remaining/length
                return tuple(a+(b-a)*fraction for a,b in zip(following, endpoint, strict=True))
            remaining -= length
            following = endpoint
        return self.points[-1]

    def condition(self, pose):
        return waypoint_instruction(*waypoint_labels(pose, self.target(pose)))


def teacher_waypoints(meta):
    start = meta["outbound_frames"]
    poses = meta["pose"]
    route = RouteLookahead(poses[start][:2], meta["scene"]["teacher_targets"])
    return [route.target(pose) for pose in poses[start:]]

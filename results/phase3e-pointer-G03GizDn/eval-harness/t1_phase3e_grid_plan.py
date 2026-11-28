"""Plan world-coordinate waypoints on the graph observed during the outbound walk."""

import math
from collections import defaultdict
from heapq import heappop, heappush
from itertools import pairwise

from t1_waypoint import LOOKAHEAD_BLOCKS, project, relative_waypoint, waypoint_instruction, waypoint_labels


TRACKING_REPAIR = "committed_route_and_heading"


def cell(point):
    return math.floor(point[0]), math.floor(point[1])


def room_observations(points):
    """Find the two deepest lateral excursions, in the room readout's slot order."""
    x, z = points[0]
    dx, dz = points[-1][0] - x, points[-1][1] - z
    length = math.hypot(dx, dz)
    if not length:
        raise ValueError("The outbound route has no start-to-end axis")
    lateral = [(a-x)*dz/length - (b-z)*dx/length for a, b in points]
    depth = [(a-x)*dx/length + (b-z)*dz/length for a, b in points]
    rooms = []
    for sign in (-1, 1):
        excursion = max(sign * value for value in lateral)
        if excursion < 2:
            raise ValueError("The outbound route did not visit both sides of the arena")
        index = max((i for i, value in enumerate(lateral) if sign * value > excursion / 2),
                    key=depth.__getitem__)
        rooms.append(points[index])
    return tuple(rooms)


def crossed_cells(start, end):
    """Identify cells intersected by a chord without rounding away corner crossings."""
    breaks = {0., 1.}
    for a, b in zip(start, end, strict=True):
        if a != b:
            for boundary in range(math.floor(min(a, b)) + 1, math.ceil(max(a, b))):
                breaks.add((boundary - a) / (b - a))
    marks = sorted(breaks)
    points = [start]
    for lo, hi in pairwise(marks):
        fraction = (lo + hi) / 2
        points.append(tuple(a + fraction * (b - a) for a, b in zip(start, end, strict=True)))
    points.append(end)
    return [key for i, key in enumerate(map(cell, points)) if not i or key != cell(points[i-1])]


class GridRoute:
    def __init__(self, outbound, slot=None, *, goal=None):
        points = [tuple(pose[:2]) for pose in outbound]
        if len(points) < 2 or (goal is None and slot not in (0, 1)):
            raise ValueError("A grid route requires an outbound walk and a room choice")
        self.goal = room_observations(points)[slot] if goal is None else tuple(goal)
        if self.goal not in points:
            raise ValueError("The goal must be a visited point")
        self.neighbours = defaultdict(set)
        observed = defaultdict(list)
        self.crossing = {}
        for point in points:
            observed[cell(point)].append(point)
        for a, b in pairwise(points):
            left, right = cell(a), cell(b)
            if left != right:
                self.neighbours[left].add(right)
                self.neighbours[right].add(left)
                self.crossing.setdefault((left, right), (a, b))
                self.crossing.setdefault((right, left), (b, a))
        self.points = {key: min(samples, key=lambda p: math.dist(p, (key[0]+.5, key[1]+.5)))
                       for key, samples in observed.items()}
        goal = cell(self.goal)
        self.points[goal] = self.goal
        self.distance = {goal: 0.}
        self.next = {}
        queue = [(0., goal)]
        while queue:
            length, current = heappop(queue)
            if length != self.distance[current]:
                continue
            for adjacent in sorted(self.neighbours[current]):
                candidate = length + math.dist(self.points[current], self.points[adjacent])
                if candidate < self.distance.get(adjacent, math.inf):
                    self.distance[adjacent] = candidate
                    self.next[adjacent] = current
                    heappush(queue, (candidate, adjacent))

    def known_chord(self, start, end):
        cells = crossed_cells(start, end)
        return all(b in self.neighbours.get(a, ()) for a, b in pairwise(cells))

    def target(self, pose):
        point = tuple(pose[:2])
        start = cell(point)
        if start not in self.distance:
            start = min(self.distance, key=lambda key: (math.dist(point, self.points[key]), key))
            # Rejoin known terrain before looking ahead, including deviations larger than two blocks.
            return self.points[start]
        if start not in self.next:
            return self.goal
        node = start
        previous = point
        left = LOOKAHEAD_BLOCKS
        target = None
        while node in self.next:
            following = self.next[node]
            waypoint = self.points[following]
            length = math.dist(previous, waypoint)
            if length > left:
                waypoint = tuple(a + (b-a)*left/length for a, b in zip(previous, waypoint, strict=True))
            if not self.known_chord(point, waypoint):
                break
            target = waypoint
            if length >= left:
                break
            left -= length
            previous = waypoint
            node = following
        if target is None:
            before, after = self.crossing[start, self.next[start]]
            return after if self.known_chord(point, after) else before
        return target

    def instruction(self, pose):
        return waypoint_instruction(*waypoint_labels(pose, self.target(pose)))


def straight_sections(points, known_chord):
    """Retain the corners of the observed path while merging nearly collinear edges."""
    kept = [points[0]]
    start = 0
    for end in range(2, len(points)):
        if not known_chord(points[start], points[end]) or any(
                math.dist(p, project(p, points[start], points[end])[1]) > .18
                for p in points[start+1:end]):
            start = end - 1
            kept.append(points[start])
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


class CommittedGridRoute(GridRoute):
    """Track one shortest path, with recovery tied to the current path segment."""

    def __init__(self, outbound, slot=None, *, goal=None):
        super().__init__(outbound, slot, goal=goal)
        self.sections = None
        self.segment = 0
        self.progress = 0.

    def target(self, pose):
        point = tuple(pose[:2])
        if self.sections is None:
            node = cell(point)
            if node not in self.distance:
                node = min(self.distance, key=lambda key: (math.dist(point, self.points[key]), key))
            points = [self.points[node]]
            while node in self.next:
                following = self.next[node]
                a, b = self.points[node], self.points[following]
                if not self.known_chord(a, b):
                    points.extend(self.crossing[node, following])
                points.append(b)
                node = following
            distinct = [points[0], *(b for a, b in pairwise(points) if a != b)]
            self.sections = straight_sections(distinct, self.known_chord)

        while self.segment < len(self.sections) - 2 and math.dist(point, self.sections[self.segment+1]) <= .42:
            self.segment += 1
            self.progress = 0.
        if len(self.sections) == 1:
            return self.goal
        start, end = self.sections[self.segment:self.segment+2]
        length = math.dist(start, end)
        fraction, projection = project(point, start, end)
        offset = math.dist(point, projection)
        if offset <= .65:
            self.progress = max(self.progress, fraction * length)
        # Aim nearer when rejoining this segment. The recovery point stays on the committed route.
        lookahead = 1. if offset > .65 else LOOKAHEAD_BLOCKS
        target_fraction = min(1., (self.progress + lookahead) / length)
        return tuple(a + (b-a)*target_fraction for a, b in zip(start, end, strict=True))


def heading_turn(pose, target):
    """Return a bounded corrective turn when motion would start at a large heading error."""
    angle, distance = relative_waypoint(pose, target)
    if abs(angle) > 15. and distance > .05:
        return math.copysign(min(10., abs(angle)), angle)
    return None

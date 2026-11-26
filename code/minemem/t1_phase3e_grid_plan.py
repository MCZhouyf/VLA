"""Plan world-coordinate waypoints on the graph observed during the outbound walk."""

import math
from collections import defaultdict
from heapq import heappop, heappush
from itertools import pairwise

from t1_waypoint import LOOKAHEAD_BLOCKS, waypoint_instruction, waypoint_labels


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
    def __init__(self, outbound, slot):
        points = [tuple(pose[:2]) for pose in outbound]
        if len(points) < 2 or slot not in (0, 1):
            raise ValueError("A grid route requires an outbound walk and a room choice")
        self.goal = room_observations(points)[slot]
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

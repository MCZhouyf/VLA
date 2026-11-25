"""Regression tests for the witnessed two-bin yaw oscillation."""

import unittest

from t1_waypoint import (STABILIZER_ALTERNATING_TURNS, STABILIZER_FORWARD_STEPS,
                         WaypointTurnStabilizer, waypoint_instruction)


def stationary_turn(yaw):
    return {"camera": (0.0, yaw), "forward": 0}


class WaypointTurnStabilizerTest(unittest.TestCase):
    def test_breaks_a_sustained_oscillation_and_limits_the_forward_burst(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                guard = WaypointTurnStabilizer()
                ahead = waypoint_instruction(0, 3)
                adjacent = waypoint_instruction(direction, 3)
                for _ in range(STABILIZER_ALTERNATING_TURNS // 2):
                    self.assertFalse(guard.should_advance(ahead, stationary_turn(-10 * direction)))
                    self.assertFalse(guard.should_advance(adjacent, stationary_turn(10 * direction)))
                for _ in range(STABILIZER_FORWARD_STEPS):
                    self.assertTrue(guard.should_advance(ahead, stationary_turn(-10 * direction)))
                self.assertFalse(guard.should_advance(ahead, stationary_turn(-10 * direction)))

    def test_leaves_normal_turns_and_advances_unchanged(self):
        guard = WaypointTurnStabilizer()
        for _ in range(20):
            self.assertFalse(guard.should_advance(waypoint_instruction(0, 3), stationary_turn(-10)))
            self.assertFalse(guard.should_advance(waypoint_instruction(2, 3), stationary_turn(10)))
            self.assertFalse(guard.should_advance(waypoint_instruction(1, 3), {"camera": (0, 10), "forward": 1}))

    def test_does_not_override_after_alignment_is_lost(self):
        guard = WaypointTurnStabilizer()
        ahead = waypoint_instruction(0, 4)
        right = waypoint_instruction(1, 4)
        for _ in range(STABILIZER_ALTERNATING_TURNS // 2):
            guard.should_advance(right, stationary_turn(10))
            guard.should_advance(ahead, stationary_turn(-10))
        self.assertFalse(guard.should_advance(right, stationary_turn(10)))
        self.assertFalse(guard.should_advance(ahead, stationary_turn(-10)))


if __name__ == "__main__":
    unittest.main()

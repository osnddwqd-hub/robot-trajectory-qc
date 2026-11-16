"""Regression tests for event attribution and paired fault injection."""
from dataclasses import replace
import unittest

import numpy as np

from robot_qc.benchmark import Scenario, event_score, inject
from robot_qc.quality import CHANNELS, PERIOD, Event, Trajectory, detect, fit_thresholds


class QualityTests(unittest.TestCase):
    def setUp(self):
        frames = np.arange(100)
        self.item = Trajectory(1, frames, frames * PERIOD, np.zeros((100, len(CHANNELS))))
        self.thresholds = np.ones(len(CHANNELS))

    def test_gap_uses_frame_span_without_timestamp_or_motion_false_alarm(self):
        keep = ~np.isin(self.item.frames, [32, 33])
        state = self.item.state.copy()
        state[34:] = 10
        trajectory = Trajectory(1, self.item.frames[keep], self.item.time[keep], state[keep])
        events = detect(trajectory, self.thresholds)
        self.assertEqual([(e.kind, e.first_frame, e.last_frame, e.observed) for e in events],
                         [("missing_frame", 32, 33, 2)])

    def test_duplicate_timestamp_keeps_input_order(self):
        time = self.item.time.copy()
        time[31] = time[30]
        events = detect(replace(self.item, time=time), self.thresholds)
        self.assertEqual([(e.kind, e.first_frame) for e in events],
                         [("timestamp", 31), ("timestamp", 32)])

    def test_spike_counts_once_and_requires_matching_channel(self):
        state = self.item.state.copy()
        state[30, 2] = 5
        events = detect(replace(self.item, state=state), self.thresholds)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].first_frame, events[0].last_frame), (30, 31))
        truth = [Event("joint_jump", 30, 30, CHANNELS[2])]
        self.assertEqual(event_score(truth, events), {"event_tp": 1, "event_fp": 0, "event_fn": 0})
        truth = [replace(truth[0], channel=CHANNELS[3])]
        self.assertEqual(event_score(truth, events), {"event_tp": 0, "event_fp": 1, "event_fn": 1})

    def test_calibration_uses_only_valid_motion_edges(self):
        time = self.item.time.copy()
        time[10] += 0.01
        state = self.item.state.copy()
        state[10, 0] = 8
        state[:, 6] = self.item.frames * 0.001
        result = fit_thresholds([replace(self.item, time=time, state=state)])
        self.assertEqual(result[0], 1e-6)
        self.assertAlmostEqual(result[6], 0.00125)

    def test_injection_does_not_mutate_reference_and_is_deterministic(self):
        scene = Scenario("sample", "random_drop", 5)
        first, truth = inject(self.item, scene, self.thresholds, np.random.default_rng(6))
        second, repeated_truth = inject(self.item, scene, self.thresholds, np.random.default_rng(6))
        np.testing.assert_array_equal(first.frames, second.frames)
        self.assertEqual(truth, repeated_truth)
        self.assertEqual(len(first.frames), 95)
        self.assertEqual(len(self.item.frames), 100)
        self.assertEqual(event_score(truth, detect(first, self.thresholds))["event_fn"], 0)


if __name__ == "__main__":
    unittest.main()

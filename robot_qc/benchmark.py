"""Inject labeled faults and score trajectory decisions and event localization."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .quality import CHANNELS, PERIOD, Event, Trajectory, detect


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: str
    strength: float
    noise: float = 0.0


SCENARIOS = (
    *(Scenario(f"random_drop_{rate:g}pct", "random_drop", rate)
      for rate in (0.5, 1, 5)),
    *(Scenario(f"burst_drop_{length}", "burst_drop", length) for length in (1, 3, 5)),
    Scenario("timestamp_duplicate", "duplicate", 1),
    Scenario("timestamp_reverse", "reverse", 1),
    Scenario("timestamp_step_3ms", "time_step", 0.003),
    Scenario("timestamp_step_10ms", "time_step", 0.010),
    *(Scenario(f"spike_{amplitude:g}x_noise_{noise:g}x", "spike", amplitude, noise)
      for noise in (0, 0.1, 0.3) for amplitude in (0.5, 1, 2)),
)


def add_noise(item: Trajectory, scale: float, thresholds: np.ndarray,
              rng: np.random.Generator) -> Trajectory:
    noise = rng.standard_normal(item.state.shape) * (scale * thresholds)
    return replace(item, state=item.state + noise)


def inject(item: Trajectory, scene: Scenario, thresholds: np.ndarray,
           rng: np.random.Generator) -> tuple[Trajectory, list[Event]]:
    # Preserve endpoints so both sides of an injected gap are observable.
    row = int(rng.integers(2, len(item.frames) - 6))
    frames = item.frames
    frame = int(frames[row])
    if scene.fault in ("random_drop", "burst_drop"):
        if scene.fault == "random_drop":
            size = max(1, round(len(frames) * scene.strength / 100))
            removed = np.sort(rng.choice(np.arange(1, len(frames) - 1), size, replace=False))
        else:
            removed = np.arange(row, row + int(scene.strength))
        keep = np.ones(len(frames), dtype=bool)
        keep[removed] = False
        groups = np.split(removed, np.flatnonzero(np.diff(removed) > 1) + 1)
        truth = [Event("missing_frame", int(frames[g[0]]), int(frames[g[-1]])) for g in groups]
        return Trajectory(item.episode, frames[keep], item.time[keep], item.state[keep]), truth

    if scene.fault == "spike":
        channel_index = int(rng.integers(len(CHANNELS)))
        sign = rng.choice((-1, 1))
        state = item.state.copy()
        state[row, channel_index] += sign * scene.strength * thresholds[channel_index]
        return replace(item, state=state), [Event("joint_jump", frame, frame,
                                                  CHANNELS[channel_index])]

    time = item.time.copy()
    if scene.fault == "time_step":
        time[row:] += scene.strength
        affected = [frame]
    else:
        time[row] = time[row - 1]
        if scene.fault == "reverse":
            time[row] -= PERIOD / 2
        affected = [frame, int(frames[row + 1])]
    return replace(item, time=time), [Event("timestamp", number, number) for number in affected]


def matches(expected: Event, actual: Event) -> bool:
    if expected.kind != actual.kind or expected.channel != actual.channel:
        return False
    if expected.kind == "joint_jump":
        # Incoming and outgoing edges around a one-frame spike refer to the same fault.
        return (actual.first_frame <= expected.first_frame + 1
                and actual.last_frame >= expected.first_frame)
    return (expected.first_frame, expected.last_frame) == (actual.first_frame, actual.last_frame)


def event_score(truth: list[Event], predicted: list[Event]) -> dict:
    remaining = list(predicted)
    hits = 0
    for expected in truth:
        match = next((index for index, event in enumerate(remaining)
                      if matches(expected, event)), None)
        if match is not None:
            remaining.pop(match)
            hits += 1
    return {"event_tp": hits, "event_fp": len(remaining), "event_fn": len(truth) - hits}


def run_benchmark(items: list[Trajectory], thresholds: np.ndarray, seed: int, repeats: int):
    case_rows, label_rows, example = [], [], None
    for scene_index, scene in enumerate(SCENARIOS):
        for repeat in range(repeats):
            for original in items:
                # Pair each injected run with the same background noise realization.
                noise_rng = np.random.default_rng([seed, repeat, original.episode])
                fault_index = 100 if scene.fault == "spike" else scene_index
                fault_rng = np.random.default_rng([seed, repeat, original.episode, fault_index])
                control = add_noise(original, scene.noise, thresholds, noise_rng)
                damaged, truth = inject(control, scene, thresholds, fault_rng)
                predicted = detect(damaged, thresholds)
                control_events = detect(control, thresholds)
                case = {"scenario": scene.name, "repeat": repeat, "episode": original.episode}
                for source, events in (("injected", truth), ("predicted", predicted),
                                       ("control", control_events)):
                    label_rows.extend({**case, "source": source, **asdict(event)} for event in events)
                case_rows.append({
                    **case,
                    "trajectory_tp": int(bool(predicted)), "trajectory_fp": int(bool(control_events)),
                    "trajectory_fn": int(not predicted), "trajectory_tn": int(not control_events),
                    **event_score(truth, predicted),
                })
                if (example is None and scene.name == "spike_2x_noise_0x"
                        and any(matches(truth[0], event) for event in predicted)):
                    example = (original, damaged, truth[0], predicted)
    cases = pd.DataFrame(case_rows)
    numeric = ["trajectory_tp", "trajectory_fp", "trajectory_fn", "trajectory_tn",
               "event_tp", "event_fp", "event_fn"]
    summary = cases.groupby("scenario", sort=False)[numeric].sum().reset_index()
    summary["trajectory_fpr"] = summary.trajectory_fp / (summary.trajectory_fp + summary.trajectory_tn)
    summary["trajectory_fnr"] = summary.trajectory_fn / (summary.trajectory_tp + summary.trajectory_fn)
    summary["event_recall"] = summary.event_tp / (summary.event_tp + summary.event_fn)
    summary["event_precision"] = summary.event_tp / (summary.event_tp + summary.event_fp).replace(0, np.nan)
    return summary, cases, pd.DataFrame(label_rows), example


def check_reference(items: list[Trajectory]) -> None:
    """Require complete, cadence-consistent references before assigning fault labels."""
    for item in items:
        if len(item.frames) < 10 or not np.array_equal(item.frames, np.arange(len(item.frames))):
            raise ValueError(f"Episode {item.episode} lacks a complete reference frame sequence")
        if not np.allclose(np.diff(item.time), PERIOD, atol=1e-4, rtol=0):
            raise ValueError(f"Episode {item.episode} has irregular reference timestamps")

"""Read ALOHA trajectories, fit thresholds, and locate quality events."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
        "waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
        "wrist_rotate", "gripper",
    )
)
PERIOD = 0.02
TIMING_TOLERANCE = 0.005
COLUMNS = ["episode_index", "frame_index", "timestamp", "observation.state"]


@dataclass(frozen=True)
class Trajectory:
    episode: int
    frames: np.ndarray
    time: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class Event:
    kind: str
    first_frame: int
    last_frame: int
    channel: str = ""
    observed: float | int | None = None
    reference: float | int | None = None
    reason: str = ""


def read_trajectories(path: Path) -> dict[int, Trajectory]:
    table = pd.read_parquet(path, columns=COLUMNS)
    if table.empty:
        raise ValueError("No trajectory rows found")
    for column in ("episode_index", "frame_index"):
        if not pd.api.types.is_integer_dtype(table[column]) or table[column].isna().any():
            raise ValueError(f"{column} must contain integer IDs")
    trajectories = {}
    for episode, rows in table.groupby("episode_index", sort=True):
        state = np.stack(rows["observation.state"]).astype(float)
        time = rows["timestamp"].to_numpy(dtype=float)
        if state.shape != (len(rows), len(CHANNELS)) or not np.isfinite(state).all():
            raise ValueError(f"Episode {episode}: expected 14 finite state values per row")
        if not np.isfinite(time).all():
            raise ValueError(f"Episode {episode}: expected finite timestamps")
        trajectories[int(episode)] = Trajectory(
            int(episode), rows["frame_index"].to_numpy(dtype=np.int64), time, state
        )
    return trajectories


def valid_motion_edges(trajectory: Trajectory) -> np.ndarray:
    return ((np.diff(trajectory.frames) == 1)
            & (np.abs(np.diff(trajectory.time) - PERIOD) <= TIMING_TOLERANCE))


def fit_thresholds(trajectories: list[Trajectory]) -> np.ndarray:
    steps = np.concatenate([
        np.abs(np.diff(item.state, axis=0))[valid_motion_edges(item)]
        for item in trajectories
    ])
    if len(steps) == 0:
        raise ValueError("Calibration needs at least one valid consecutive sample pair")
    # A robust empirical upper tail per channel, including separate gripper thresholds.
    return np.maximum(1e-6, 1.25 * np.quantile(steps, 0.999, axis=0))


def detect(trajectory: Trajectory, thresholds: np.ndarray) -> list[Event]:
    frames = trajectory.frames
    frame_steps = np.diff(frames)
    time_steps = np.diff(trajectory.time)
    events = []

    for edge, (frame_step, time_step) in enumerate(zip(frame_steps, time_steps)):
        previous, current = map(int, frames[edge:edge + 2])
        if frame_step > 1:
            events.append(Event("missing_frame", previous + 1, current - 1,
                                observed=int(frame_step - 1), reference=0,
                                reason="gap in original frame index"))
        if frame_step <= 0:
            events.append(Event("frame_order", current, current, observed=int(frame_step),
                                reference=1, reason="duplicate or reversed frame index"))
        expected = float(frame_step * PERIOD) if frame_step > 0 else None
        if time_step <= 0 or (expected is not None
                              and abs(time_step - expected) > TIMING_TOLERANCE):
            reason = "non-increasing timestamp" if time_step <= 0 else "unexpected interval"
            events.append(Event("timestamp", current, current, observed=float(time_step),
                                reference=expected, reason=reason))

    motion = np.abs(np.diff(trajectory.state, axis=0))
    flagged = (motion > thresholds) & valid_motion_edges(trajectory)[:, None]
    for channel_index, channel in enumerate(CHANNELS):
        edges = np.flatnonzero(flagged[:, channel_index])
        if not len(edges):
            continue
        for group in np.split(edges, np.flatnonzero(np.diff(edges) > 1) + 1):
            first, last = int(group[0]), int(group[-1])
            events.append(Event(
                "joint_jump", int(frames[first + 1]), int(frames[last + 1]), channel,
                float(motion[group, channel_index].max()), float(thresholds[channel_index]),
                "consecutive position changes above channel threshold",
            ))
    return sorted(events, key=lambda e: (e.first_frame, e.last_frame, e.kind, e.channel))

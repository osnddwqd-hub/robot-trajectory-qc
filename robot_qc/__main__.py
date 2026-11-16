"""Download the pinned reference data and generate a reproducible QC report."""
import argparse
from collections import Counter
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from .benchmark import check_reference, run_benchmark
from .quality import CHANNELS, Event, detect, fit_thresholds, read_trajectories
from .report import plot_benchmark, plot_episode, plot_example, write_report

DATA_URL = ("https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted"
            "/resolve/c5467ba9059e4c65f27775ced212076980a173fb/data/chunk-000/file-000.parquet")
DATA_HASH = "4c74dc47dd06f33105116031d9024abaa1080847dbff99629be01a2d6367e441"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(path: Path) -> None:
    if path.is_file() and sha256(path) == DATA_HASH:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=90) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != DATA_HASH:
        raise ValueError("Downloaded parquet checksum differs from pinned dataset")
    pending = path.with_suffix(".partial")
    pending.write_bytes(data)
    pending.replace(path)


def write_quality(trajectories, thresholds, out: Path, split: dict):
    quality_rows, event_rows = [], []
    for number, item in trajectories.items():
        events = detect(item, thresholds)
        counts = Counter(event.kind for event in events)
        quality_rows.append({
            "episode": number, "split": split.get(number, "input"),
            "frames": len(item.frames), "passed": not events, "events": len(events),
            "missing_frames": sum(event.observed for event in events if event.kind == "missing_frame"),
            "timestamp": counts["timestamp"], "joint_jump": counts["joint_jump"],
            "frame_order": counts["frame_order"],
        })
        event_rows.extend({"episode": number, **asdict(event)} for event in events)
        plot_episode(item, events, out / "episodes" / f"episode_{number}.png")
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(out / "quality_summary.csv", index=False)
    pd.DataFrame(event_rows, columns=["episode", *(field.name for field in fields(Event))]).to_csv(
        out / "anomalies.csv", index=False,
    )

    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/aloha.parquet"))
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--download", action="store_true", help="Fetch the pinned ALOHA parquet")
    parser.add_argument("--thresholds-from", type=Path,
                        help="Reuse run.json thresholds for QC only on compatible input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.seed < 0 or args.repeats < 1:
        parser.error("seed must be nonnegative and repeats must be at least 1")
    if args.download:
        download(args.input)
    if not args.input.is_file():
        parser.error("input file does not exist; supply --input or --download")

    trajectories = read_trajectories(args.input)
    out = args.out
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    if args.thresholds_from:
        stored = json.loads(args.thresholds_from.read_text(encoding="utf-8"))["thresholds"]
        thresholds = np.array([stored[channel] for channel in CHANNELS], dtype=float)
        if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
            parser.error("stored thresholds must be finite and positive")
        quality = write_quality(trajectories, thresholds, out, {})
        write_report(out, quality)
        print(f"{int(quality.passed.sum())}/{len(quality)} episodes passed. Report: {out / 'report.html'}")
        return
    if len(trajectories) <= 20:
        parser.error("need at least 21 trajectories for a 20-episode calibration split")
    ids = np.random.default_rng(args.seed).permutation(list(trajectories)).tolist()
    calibration, test = sorted(ids[:20]), sorted(ids[20:])
    thresholds = fit_thresholds([trajectories[number] for number in calibration])
    test_items = [trajectories[number] for number in test]
    check_reference(test_items)

    (out / "run.json").write_text(json.dumps({
        "input_sha256": sha256(args.input), "seed": args.seed, "repeats": args.repeats,
        "calibration_episodes": calibration, "test_episodes": test,
        "thresholds": dict(zip(CHANNELS, map(float, thresholds))),
    }, indent=2) + "\n", encoding="utf-8")

    split = {**dict.fromkeys(calibration, "calibration"), **dict.fromkeys(test, "test")}
    quality = write_quality(trajectories, thresholds, out, split)

    metrics, cases, labels, example = run_benchmark(test_items, thresholds, args.seed, args.repeats)
    metrics.to_csv(out / "benchmark_summary.csv", index=False)
    cases.to_csv(out / "benchmark_cases.csv", index=False)
    labels.to_csv(out / "benchmark_events.csv", index=False)
    plot_benchmark(metrics, out / "benchmark.png")
    if example is not None:
        plot_example(example, out / "example.png")
    write_report(out, quality, metrics, args.seed, args.repeats, len(calibration), len(test), example is not None)
    print(f"{int(quality.passed.sum())}/{len(quality)} raw episodes passed. Report: {out / 'report.html'}")


if __name__ == "__main__":
    main()

"""Render the benchmark and trajectory inspection views."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .quality import CHANNELS, Event, Trajectory


def plot_episode(trajectory: Trajectory, events: list[Event], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    for axis, offset, side in zip(axes, (0, 7), ("Left", "Right")):
        for index in range(offset, offset + 7):
            axis.plot(trajectory.frames, trajectory.state[:, index], lw=0.9,
                      label=CHANNELS[index].split("_", 1)[1])
        for event in events:
            if event.kind == "joint_jump" and not event.channel.startswith(side.lower()):
                continue
            color = {"joint_jump": "#c23841", "missing_frame": "#8655ac",
                     "timestamp": "#e1812c"}.get(event.kind, "#414c58")
            axis.axvspan(event.first_frame - 0.4, event.last_frame + 0.4,
                         color=color, alpha=0.22, lw=0)
        axis.set_ylabel(f"{side} state")
        axis.legend(ncol=4, fontsize=7, loc="upper right")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Original frame index")
    fig.suptitle(f"Episode {trajectory.episode:02d} | {len(events)} detected events")
    fig.savefig(path, dpi=115)
    plt.close(fig)


def plot_example(example, path: Path) -> None:
    original, injected, truth, predicted = example
    index = CHANNELS.index(truth.channel)
    center = truth.first_frame
    window = slice(max(0, center - 20), center + 21)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, sharey=True,
                             layout="constrained")
    panels = ((original, "Original"), (injected, "Injected"), (injected, "QC output"))
    for axis, (series, label) in zip(axes, panels):
        axis.plot(series.frames[window], series.state[window, index], color="#176d91", lw=1.6)
        axis.set_ylabel("State")
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold")
        axis.grid(alpha=0.2)
    axes[1].axvline(center, color="#8253b4", ls="--", label=f"Injected frame {center}")
    for event in predicted:
        if event.channel == truth.channel and event.last_frame >= center - 20 and event.first_frame <= center + 20:
            axes[2].axvspan(max(center - 20, event.first_frame) - 0.4,
                            min(center + 20, event.last_frame) + 0.4, color="#c23841", alpha=0.3)
    axes[1].legend(loc="best")
    axes[-1].set_xlabel("Original frame index")
    fig.suptitle(f"Episode {original.episode:02d} | {truth.channel} | single-frame spike")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_benchmark(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout="constrained")
    y = np.arange(len(summary))
    axes[0].barh(y, 100 * summary.event_recall, color="#176d91")
    axes[0].scatter(100 * summary.event_recall, y, color="#176d91", s=14, zorder=3)
    axes[0].set_yticks(y, summary.scenario.str.replace("_", " "))
    axes[0].invert_yaxis()
    axes[0].set_title("Event recall")
    axes[1].scatter(100 * summary.trajectory_fpr, y + .14, color="#b45c24", label="False positive")
    axes[1].scatter(100 * summary.trajectory_fnr, y - .14, color="#6240a2", label="False negative")
    axes[1].set_title("Trajectory error rates")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    for axis in axes:
        axis.set_xlim(-3, 103)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Percent")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(out: Path, quality: pd.DataFrame, metrics: pd.DataFrame | None = None,
                 seed: int = 0, repeats: int = 0, calibration_size: int = 0,
                 test_size: int = 0, has_example: bool = False) -> None:
    quality_links = quality.copy()
    quality_links["轨迹图"] = [
        f'<a href="episodes/episode_{episode}.png">打开</a>' for episode in quality.episode
    ]
    quality_table = quality_links.to_html(index=False, escape=False, border=0)
    benchmark_section = ""
    if metrics is not None:
        metrics_table = metrics.to_html(index=False, float_format=lambda x: f"{x:.3f}",
                                        na_rep="—", border=0)
        example_section = ""
        if has_example:
            example_section = """<section><h2>关节突跳定位示例</h2>
<img src="example.png" alt="原始、注入和定位结果三图对照"></section>"""
        benchmark_section = f"""
<p>校准 {calibration_size} 条，评测 {test_size} 条，随机种子 {seed}，每场景重复 {repeats} 次。</p>
<p><a href="benchmark_summary.csv">评测汇总</a> ·
<a href="benchmark_cases.csv">逐次评测</a> · <a href="benchmark_events.csv">注入与预测事件</a></p>
{example_section}
<section><h2>故障注入评测</h2>
<p>轨迹级错误率使用同噪声对照。事件级召回率要求同时命中类型和位置。
原始轨迹的自然动作也可能触发规则，结果适用于本仿真批次。</p>
<img src="benchmark.png" alt="事件召回率与轨迹误报、漏报率"><br>{metrics_table}</section>"""
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>机器人轨迹质检与评测</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{font: 16px/1.55 system-ui,sans-serif; max-width: 1100px; margin: 2rem auto;
padding: 0 1rem; color: #172838; background: #f8fafb}}
section {{overflow-x:auto; margin-bottom:2rem}}
table {{border-collapse:collapse; width:100%; background:white; white-space:nowrap}}
th,td {{text-align:right; padding:.45rem .65rem; border-bottom:1px solid #dfe6eb}}
img {{width:100%; max-width:980px; height:auto; display:block}}
a {{color:#176d91}}
</style>
<h1>机器人轨迹质检与评测</h1>
<p>输入 {len(quality)} 条轨迹，规则合格率 {quality.passed.mean():.1%}。</p>
<p><a href="anomalies.csv">异常明细</a> · <a href="quality_summary.csv">逐条质检结果</a></p>
{benchmark_section}
<section><h2>逐条轨迹</h2><p>红色为突跳，紫色为丢帧，橙色为时间戳异常。</p>
{quality_table}</section></html>"""
    (out / "report.html").write_text(document, encoding="utf-8")

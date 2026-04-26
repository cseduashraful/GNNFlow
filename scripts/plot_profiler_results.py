#!/usr/bin/env python3

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to generate profiler plots. "
        "Install it with `pip install matplotlib`."
    ) from exc

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_DIR = REPO_ROOT / "profiles"
DEFAULT_OUTPUT_DIR = DEFAULT_PROFILES_DIR / "plots"

TRAIN_STAGE_ORDER = [
    ("sampling", "Sampling"),
    ("feature_fetch", "Feature Fetch"),
    ("memory_fetch", "Memory Fetch"),
    ("memory_update", "Memory Update"),
    ("memory_write_back", "Memory Write Back"),
    ("model_forward", "Model Forward"),
    ("loss_backward_optimizer", "Loss+Backward+Opt"),
]

SETUP_STAGE_ORDER = [
    ("dataset_load_sec", "Dataset Load"),
    ("graph_build_sec", "Graph Build"),
    ("graph_ingestion_sec", "Graph Ingestion"),
    ("feature_load_sec", "Feature Load"),
    ("model_init_sec", "Model Init"),
    ("cache_init_sec", "Cache Init"),
]

TRAIN_STAGE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
]

SETUP_STAGE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
]

MODEL_MARKERS = {
    "TGN": "o",
    "TGAT": "^",
    "DySAT": "s",
    "GRAPHSAGE": "D",
    "GAT": "P",
}

MODEL_COLORS = {
    "TGN": "#1f77b4",
    "TGAT": "#ff7f0e",
    "DySAT": "#2ca02c",
    "GRAPHSAGE": "#d62728",
    "GAT": "#9467bd",
}


@dataclass
class ProfileRun:
    summary_path: Path
    run_dir: Path
    source_kind: str
    modified_at: float
    model: str
    dataset: str
    cache: str
    edge_cache_ratio: float
    node_cache_ratio: float
    snapshot_time_window: float
    ingestion_batch_size: int
    batch_size: int
    world_size: int
    step_time_avg_sec: Optional[float]
    throughput_avg: Optional[float]
    peak_allocated_bytes: Optional[float]
    peak_reserved_bytes: Optional[float]
    gpu_load_avg_pct: Optional[float]
    gpu_memory_util_avg_pct: Optional[float]
    gpu_memory_used_avg_mb: Optional[float]
    cpu_max_rss_bytes: Optional[float]
    train_stage_time_sec: Dict[str, Optional[float]]
    setup_time_sec: Dict[str, Optional[float]]

    def config_key(self) -> Tuple[Any, ...]:
        return (
            self.model,
            self.dataset,
            self.cache,
            self.batch_size,
            self.world_size,
            self.edge_cache_ratio,
            self.node_cache_ratio,
            self.snapshot_time_window,
            self.ingestion_batch_size,
        )

    def run_label(self) -> str:
        parts = [
            self.model,
            self.dataset,
            f"bs={self.batch_size}",
            self.cache,
            f"ws={self.world_size}",
        ]
        if self.edge_cache_ratio != 0 or self.node_cache_ratio != 0:
            parts.append(f"e={self.edge_cache_ratio:g}/n={self.node_cache_ratio:g}")
        if self.snapshot_time_window != 0:
            parts.append(f"tw={self.snapshot_time_window:g}")
        return " | ".join(parts)

    def wrapped_run_label(self, width: int = 34) -> str:
        return fill(self.run_label(), width=width)

    def batch_group_label(self) -> str:
        return (
            f"{self.model} | {self.dataset} | {self.cache} | "
            f"ws={self.world_size} | edge={self.edge_cache_ratio:g} | "
            f"node={self.node_cache_ratio:g}"
        )


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#4E79A7")


def shared_run_value(runs: Sequence[ProfileRun], accessor) -> Any:
    values = {accessor(run) for run in runs}
    if len(values) != 1:
        return None
    return next(iter(values))


def build_overview_context(runs: Sequence[ProfileRun]) -> Dict[str, Any]:
    return {
        "dataset": shared_run_value(runs, lambda run: run.dataset),
        "cache": shared_run_value(runs, lambda run: run.cache),
        "world_size": shared_run_value(runs, lambda run: run.world_size),
        "edge_cache_ratio": shared_run_value(runs, lambda run: run.edge_cache_ratio),
        "node_cache_ratio": shared_run_value(runs, lambda run: run.node_cache_ratio),
        "snapshot_time_window": shared_run_value(
            runs, lambda run: run.snapshot_time_window),
    }


def overview_context_subtitle(shared_context: Dict[str, Any]) -> str:
    parts: List[str] = []
    if shared_context["dataset"] is not None:
        parts.append(str(shared_context["dataset"]))
    if shared_context["cache"] is not None:
        parts.append(str(shared_context["cache"]))
    if shared_context["world_size"] is not None:
        parts.append(f"ws={shared_context['world_size']}")
    if shared_context["edge_cache_ratio"] is not None and \
            shared_context["node_cache_ratio"] is not None:
        parts.append(
            f"e={shared_context['edge_cache_ratio']:g}/n="
            f"{shared_context['node_cache_ratio']:g}"
        )
    if shared_context["snapshot_time_window"] not in (None, 0):
        parts.append(f"tw={shared_context['snapshot_time_window']:g}")
    return " | ".join(parts)


def overview_run_label(run: ProfileRun, shared_context: Dict[str, Any]) -> str:
    primary = f"{run.model} | bs={run.batch_size}"
    extras: List[str] = []

    if shared_context["dataset"] is None:
        extras.append(run.dataset)
    if shared_context["cache"] is None:
        extras.append(run.cache)
    if shared_context["world_size"] is None:
        extras.append(f"ws={run.world_size}")
    if shared_context["edge_cache_ratio"] is None or \
            shared_context["node_cache_ratio"] is None:
        extras.append(f"e={run.edge_cache_ratio:g}/n={run.node_cache_ratio:g}")
    if shared_context["snapshot_time_window"] is None and \
            run.snapshot_time_window != 0:
        extras.append(f"tw={run.snapshot_time_window:g}")

    if len(extras) == 0:
        return primary
    return primary + "\n" + fill(" | ".join(extras), width=24)


def can_use_grouped_overview(runs: Sequence[ProfileRun]) -> bool:
    if len(runs) == 0:
        return False

    shared_context = build_overview_context(runs)
    required_shared_keys = [
        "dataset",
        "cache",
        "world_size",
        "edge_cache_ratio",
        "node_cache_ratio",
        "snapshot_time_window",
    ]
    if any(shared_context[key] is None for key in required_shared_keys):
        return False

    seen_pairs = set()
    for run in runs:
        pair = (run.model, run.batch_size)
        if pair in seen_pairs:
            return False
        seen_pairs.add(pair)

    return len({run.model for run in runs}) >= 2


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_avg(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        return as_float(value.get("avg"))
    return as_float(value)


def metric_max(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        return as_float(value.get("max"))
    return as_float(value)


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) == 0:
        return None
    return sum(numeric_values) / len(numeric_values)


def maximum(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) == 0:
        return None
    return max(numeric_values)


def discover_summary_paths(profiles_dir: Path) -> List[Path]:
    if not profiles_dir.exists():
        raise FileNotFoundError(f"Profiles directory does not exist: {profiles_dir}")

    summary_paths: List[Path] = []
    for run_dir in sorted(path for path in profiles_dir.iterdir() if path.is_dir()):
        aggregate_summary = run_dir / "summary_all_ranks.json"
        rank0_summary = run_dir / "rank0" / "summary.json"
        if aggregate_summary.exists():
            summary_paths.append(aggregate_summary)
        elif rank0_summary.exists():
            summary_paths.append(rank0_summary)
    return summary_paths


def build_profile_run(summary_path: Path) -> ProfileRun:
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    rank_summaries = summary.get("ranks")
    if not isinstance(rank_summaries, list) or len(rank_summaries) == 0:
        rank_summaries = [summary]

    train_stage_time_sec = {
        key: metric_avg(nested_get(summary, "stage_time_sec", key))
        if nested_get(summary, "stage_time_sec", key) is not None
        else average(
            metric_avg(nested_get(rank_summary, "stage_time_sec", key))
            for rank_summary in rank_summaries
        )
        for key, _ in TRAIN_STAGE_ORDER
    }

    setup_time_sec = {
        key: metric_avg(nested_get(summary, "setup_metrics_sec", key))
        if nested_get(summary, "setup_metrics_sec", key) is not None
        else average(
            metric_avg(nested_get(rank_summary, "setup_metrics_sec", key))
            for rank_summary in rank_summaries
        )
        for key, _ in SETUP_STAGE_ORDER
    }

    gpu_memory_used_avg_mb = average(
        as_float(nested_get(rank_summary, "gpu_monitor", "avg_memory_used_mb"))
        for rank_summary in rank_summaries
    )
    cpu_max_rss_bytes = maximum(
        as_float(nested_get(rank_summary, "cpu_memory", "max_rss_bytes"))
        for rank_summary in rank_summaries
    )

    peak_allocated_bytes = metric_max(summary.get("peak_allocated_bytes"))
    if peak_allocated_bytes is None:
        peak_allocated_bytes = maximum(
            as_float(nested_get(rank_summary, "cuda_memory", "peak_allocated_bytes"))
            for rank_summary in rank_summaries
        )

    peak_reserved_bytes = metric_max(summary.get("peak_reserved_bytes"))
    if peak_reserved_bytes is None:
        peak_reserved_bytes = maximum(
            as_float(nested_get(rank_summary, "cuda_memory", "peak_reserved_bytes"))
            for rank_summary in rank_summaries
        )

    gpu_load_avg_pct = metric_avg(summary.get("gpu_load_pct"))
    if gpu_load_avg_pct is None:
        gpu_load_avg_pct = average(
            as_float(nested_get(rank_summary, "gpu_monitor", "avg_load_pct"))
            for rank_summary in rank_summaries
        )

    gpu_memory_util_avg_pct = metric_avg(summary.get("gpu_memory_util_pct"))
    if gpu_memory_util_avg_pct is None:
        gpu_memory_util_avg_pct = average(
            as_float(nested_get(rank_summary, "gpu_monitor", "avg_memory_util_pct"))
            for rank_summary in rank_summaries
        )

    return ProfileRun(
        summary_path=summary_path,
        run_dir=summary_path.parent if summary_path.name == "summary_all_ranks.json"
        else summary_path.parents[1],
        source_kind="aggregate" if summary_path.name == "summary_all_ranks.json"
        else "rank0",
        modified_at=summary_path.stat().st_mtime,
        model=str(summary["model"]),
        dataset=str(summary["dataset"]),
        cache=str(summary.get("cache", "unknown")),
        edge_cache_ratio=float(summary.get("edge_cache_ratio", 0.0)),
        node_cache_ratio=float(summary.get("node_cache_ratio", 0.0)),
        snapshot_time_window=float(summary.get("snapshot_time_window", 0.0)),
        ingestion_batch_size=int(summary.get("ingestion_batch_size", 0)),
        batch_size=int(summary["batch_size"]),
        world_size=int(summary.get("world_size", 1)),
        step_time_avg_sec=metric_avg(summary.get("step_time_sec")),
        throughput_avg=metric_avg(summary.get("throughput_samples_per_sec")),
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        gpu_load_avg_pct=gpu_load_avg_pct,
        gpu_memory_util_avg_pct=gpu_memory_util_avg_pct,
        gpu_memory_used_avg_mb=gpu_memory_used_avg_mb,
        cpu_max_rss_bytes=cpu_max_rss_bytes,
        train_stage_time_sec=train_stage_time_sec,
        setup_time_sec=setup_time_sec,
    )


def filter_runs(runs: Sequence[ProfileRun], args: argparse.Namespace) -> List[ProfileRun]:
    def matches(run: ProfileRun) -> bool:
        if args.models and run.model not in args.models:
            return False
        if args.datasets and run.dataset not in args.datasets:
            return False
        if args.batch_sizes and run.batch_size not in args.batch_sizes:
            return False
        if args.caches and run.cache not in args.caches:
            return False
        if args.world_sizes and run.world_size not in args.world_sizes:
            return False
        if args.edge_cache_ratios and run.edge_cache_ratio not in args.edge_cache_ratios:
            return False
        if args.node_cache_ratios and run.node_cache_ratio not in args.node_cache_ratios:
            return False
        if args.snapshot_time_windows and \
                run.snapshot_time_window not in args.snapshot_time_windows:
            return False
        return True

    return [run for run in runs if matches(run)]


def latest_per_config(runs: Sequence[ProfileRun]) -> List[ProfileRun]:
    selected: Dict[Tuple[Any, ...], ProfileRun] = {}
    for run in runs:
        key = run.config_key()
        if key not in selected or run.modified_at > selected[key].modified_at:
            selected[key] = run
    return sorted(selected.values(), key=sort_runs_key)


def sort_runs_key(run: ProfileRun) -> Tuple[Any, ...]:
    return (
        run.dataset,
        run.model,
        run.batch_size,
        run.cache,
        run.world_size,
        run.edge_cache_ratio,
        run.node_cache_ratio,
        run.snapshot_time_window,
        run.modified_at,
    )


def slugify(value: Any) -> str:
    return "".join(
        character if character.isalnum() or character in ("-", "_", ".")
        else "-"
        for character in str(value)
    )


def format_values_for_slug(values: Sequence[Any]) -> str:
    return "-".join(slugify(value) for value in values)


def build_output_prefix(args: argparse.Namespace, runs: Sequence[ProfileRun]) -> str:
    if args.output_prefix:
        return args.output_prefix

    models = args.models or sorted({run.model for run in runs})
    datasets = args.datasets or sorted({run.dataset for run in runs})
    batch_sizes = args.batch_sizes or sorted({run.batch_size for run in runs})
    caches = args.caches or sorted({run.cache for run in runs})
    world_sizes = args.world_sizes or sorted({run.world_size for run in runs})
    edge_cache_ratios = args.edge_cache_ratios or sorted(
        {run.edge_cache_ratio for run in runs})
    node_cache_ratios = args.node_cache_ratios or sorted(
        {run.node_cache_ratio for run in runs})
    snapshot_time_windows = args.snapshot_time_windows or sorted(
        {run.snapshot_time_window for run in runs})

    parts = [
        f"models-{format_values_for_slug(models)}",
        f"datasets-{format_values_for_slug(datasets)}",
        f"batches-{format_values_for_slug(batch_sizes)}",
    ]

    if len(caches) == 1:
        parts.append(f"cache-{format_values_for_slug(caches)}")
    else:
        parts.append(f"caches-{format_values_for_slug(caches)}")

    if len(world_sizes) == 1:
        parts.append(f"ws-{format_values_for_slug(world_sizes)}")
    else:
        parts.append(f"world-sizes-{format_values_for_slug(world_sizes)}")

    if len(edge_cache_ratios) > 1 or args.edge_cache_ratios:
        parts.append(f"edge-cache-{format_values_for_slug(edge_cache_ratios)}")

    if len(node_cache_ratios) > 1 or args.node_cache_ratios:
        parts.append(f"node-cache-{format_values_for_slug(node_cache_ratios)}")

    if len(snapshot_time_windows) > 1 or args.snapshot_time_windows:
        parts.append(f"time-window-{format_values_for_slug(snapshot_time_windows)}")

    return "__".join(parts)


def maybe_gib(value_bytes: Optional[float]) -> Optional[float]:
    if value_bytes is None:
        return None
    return float(value_bytes) / (1024 ** 3)


def metric_values(runs: Sequence[ProfileRun], accessor) -> List[Optional[float]]:
    return [accessor(run) for run in runs]


def plot_metric_bars(ax, runs: Sequence[ProfileRun], title: str, ylabel: str,
                     values: Sequence[Optional[float]], colors,
                     show_y_labels: bool = True):
    y_positions = list(range(len(runs)))
    heights = [0.0 if value is None else float(value) for value in values]
    if isinstance(colors, str):
        bar_colors = [colors] * len(runs)
    else:
        bar_colors = list(colors)
    ax.barh(y_positions, heights, color=bar_colors, height=0.68)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_yticks(y_positions)
    if show_y_labels:
        ax.set_yticklabels([run.wrapped_run_label() for run in runs])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.invert_yaxis()
    ax.margins(y=0.03)
    ax.xaxis.grid(True, alpha=0.25)
    ax.yaxis.grid(False)
    for y_position, plotted_value, raw_value in zip(y_positions, heights, values):
        if raw_value is None:
            ax.text(0, y_position, "n/a", va="center", ha="left", fontsize=8)
            continue
        if plotted_value == 0:
            continue
        offset = max(plotted_value * 0.01, 0.02)
        label = f"{raw_value:.2f}" if abs(raw_value) < 100 else f"{raw_value:.1f}"
        ax.text(plotted_value + offset, y_position, label,
                va="center", ha="left", fontsize=8)


def plot_overview_dashboard_grouped(runs: Sequence[ProfileRun], output_path: Path):
    metric_specs = [
        ("Avg Step Time", "ms",
         lambda run: None if run.step_time_avg_sec is None
         else run.step_time_avg_sec * 1000.0),
        ("Throughput", "samples/s", lambda run: run.throughput_avg),
        ("Peak Allocated", "GiB", lambda run: maybe_gib(run.peak_allocated_bytes)),
        ("Peak Reserved", "GiB", lambda run: maybe_gib(run.peak_reserved_bytes)),
        ("Avg GPU Load", "%", lambda run: run.gpu_load_avg_pct),
        ("Avg GPU Mem Util", "%", lambda run: run.gpu_memory_util_avg_pct),
        ("Avg GPU Mem Used", "MB", lambda run: run.gpu_memory_used_avg_mb),
        ("Peak RSS", "GiB", lambda run: maybe_gib(run.cpu_max_rss_bytes)),
    ]

    shared_context = build_overview_context(runs)
    models = list(dict.fromkeys(run.model for run in runs))
    batch_sizes = sorted({run.batch_size for run in runs})
    run_lookup = {(run.model, run.batch_size): run for run in runs}

    figure_height = max(6.8, len(batch_sizes) * 0.84)
    figure, axes = plt.subplots(
        2, 4, figsize=(19.5, figure_height), sharey=True, constrained_layout=False)
    axes_list = list(axes.flat)
    group_centers = list(range(len(batch_sizes)))
    group_height = 0.78
    bar_height = group_height / max(1, len(models))

    for axis_index, (axis, (title, ylabel, accessor)) in enumerate(zip(axes_list, metric_specs)):
        for model_index, model in enumerate(models):
            offset = -group_height / 2 + (model_index + 0.5) * bar_height
            positions: List[float] = []
            values: List[float] = []
            for center, batch_size in zip(group_centers, batch_sizes):
                run = run_lookup.get((model, batch_size))
                raw_value = None if run is None else accessor(run)
                if raw_value is None:
                    continue
                positions.append(center + offset)
                values.append(float(raw_value))

            if len(values) == 0:
                continue

            axis.barh(
                positions,
                values,
                height=bar_height * 0.86,
                color=model_color(model),
                alpha=0.95,
            )

        axis.set_title(title)
        axis.set_xlabel(ylabel)
        axis.set_yticks(group_centers)
        if axis_index % 4 == 0:
            axis.set_yticklabels([f"bs={batch_size}" for batch_size in batch_sizes])
            axis.tick_params(axis="y", labelsize=9)
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", left=False, labelleft=False)
        axis.invert_yaxis()
        axis.margins(y=0.06)
        axis.xaxis.grid(True, alpha=0.25)
        axis.yaxis.grid(False)

    legend_handles = [
        Patch(facecolor=model_color(model), label=model)
        for model in models
    ]
    subtitle = overview_context_subtitle(shared_context)
    if subtitle != "":
        subtitle = subtitle + " | grouped by batch size"
    else:
        subtitle = "Grouped by batch size"

    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.944),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.6,
    )
    figure.text(0.5, 0.968, subtitle, ha="center", va="top", fontsize=10)
    figure.suptitle("Profiler Overview", fontsize=16, y=0.992)
    figure.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.07,
                           wspace=0.15, hspace=0.26)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_overview_dashboard_fallback(runs: Sequence[ProfileRun], output_path: Path):
    metric_specs = [
        ("Avg Step Time", "ms",
         metric_values(runs, lambda run: None if run.step_time_avg_sec is None
                       else run.step_time_avg_sec * 1000.0)),
        ("Throughput", "samples/s",
         metric_values(runs, lambda run: run.throughput_avg)),
        ("Peak Allocated", "GiB",
         metric_values(runs, lambda run: maybe_gib(run.peak_allocated_bytes))),
        ("Peak Reserved", "GiB",
         metric_values(runs, lambda run: maybe_gib(run.peak_reserved_bytes))),
        ("Avg GPU Load", "%",
         metric_values(runs, lambda run: run.gpu_load_avg_pct)),
        ("Avg GPU Mem Util", "%",
         metric_values(runs, lambda run: run.gpu_memory_util_avg_pct)),
        ("Avg GPU Mem Used", "MB",
         metric_values(runs, lambda run: run.gpu_memory_used_avg_mb)),
        ("Peak RSS", "GiB",
         metric_values(runs, lambda run: maybe_gib(run.cpu_max_rss_bytes))),
    ]

    shared_context = build_overview_context(runs)
    overview_colors = [model_color(run.model) for run in runs]

    figure_height = max(6.6, len(runs) * 0.56)
    figure = plt.figure(figsize=(20.2, figure_height))
    grid = figure.add_gridspec(
        2, 5, width_ratios=[2.1, 2.8, 2.7, 2.7, 2.7], wspace=0.14, hspace=0.24)
    label_axis = figure.add_subplot(grid[:, 0])
    axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2], sharey=label_axis),
        figure.add_subplot(grid[0, 3], sharey=label_axis),
        figure.add_subplot(grid[0, 4], sharey=label_axis),
        figure.add_subplot(grid[1, 1], sharey=label_axis),
        figure.add_subplot(grid[1, 2], sharey=label_axis),
        figure.add_subplot(grid[1, 3], sharey=label_axis),
        figure.add_subplot(grid[1, 4], sharey=label_axis),
    ]

    y_positions = list(range(len(runs)))
    label_axis.set_xlim(0, 1)
    label_axis.set_ylim(-0.5, len(runs) - 0.5)
    label_axis.invert_yaxis()
    label_axis.axis("off")
    label_axis.set_title("Run Config", loc="left", pad=6)
    for y_position, run in zip(y_positions, runs):
        label_axis.text(
            0.0, y_position, overview_run_label(run, shared_context),
            va="center", ha="left", fontsize=9, color=model_color(run.model),
            fontweight="semibold")

    for axis, (title, ylabel, values) in zip(axes, metric_specs):
        plot_metric_bars(axis, runs, title, ylabel, values, overview_colors,
                         show_y_labels=False)

    subtitle = overview_context_subtitle(shared_context)
    title_y = 0.985
    top_margin = 0.89
    if subtitle != "":
        figure.text(0.53, 0.955, subtitle, ha="center", va="top", fontsize=10)
        top_margin = 0.875

    figure.suptitle("Profiler Overview", fontsize=16, y=title_y)
    figure.subplots_adjust(left=0.035, right=0.995, top=top_margin, bottom=0.06)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_overview_dashboard(runs: Sequence[ProfileRun], output_path: Path):
    if can_use_grouped_overview(runs):
        plot_overview_dashboard_grouped(runs, output_path)
        return
    plot_overview_dashboard_fallback(runs, output_path)


def plot_stacked_breakdown(runs: Sequence[ProfileRun], output_path: Path,
                           stage_order: Sequence[Tuple[str, str]],
                           colors: Sequence[str], title: str,
                           value_getter, ylabel: str,
                           normalize: bool = False):
    y_positions = list(range(len(runs)))
    bottoms = [0.0] * len(runs)
    figure_height = max(5.5, len(runs) * 0.6)
    figure, axis = plt.subplots(figsize=(16, figure_height))

    for color, (stage_key, stage_label) in zip(colors, stage_order):
        raw_values = [value_getter(run, stage_key) for run in runs]
        if normalize:
            normalized_values = []
            for run, raw_value in zip(runs, raw_values):
                total = sum(
                    value_getter(run, key) or 0.0
                    for key, _ in stage_order
                )
                if total <= 0 or raw_value is None:
                    normalized_values.append(0.0)
                else:
                    normalized_values.append(100.0 * raw_value / total)
            plot_values = normalized_values
        else:
            plot_values = [0.0 if value is None else float(value) for value in raw_values]

        axis.barh(y_positions, plot_values, left=bottoms,
                  color=color, label=stage_label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, plot_values)]

    axis.set_title(title)
    axis.set_xlabel(ylabel)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([run.wrapped_run_label() for run in runs])
    axis.invert_yaxis()
    axis.xaxis.grid(True, alpha=0.25)
    axis.yaxis.grid(False)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_batch_scaling(runs: Sequence[ProfileRun], output_path: Path):
    grouped_runs: Dict[Tuple[Any, ...], List[ProfileRun]] = {}
    for run in runs:
        key = (
            run.model,
            run.dataset,
            run.cache,
            run.world_size,
            run.edge_cache_ratio,
            run.node_cache_ratio,
            run.snapshot_time_window,
        )
        grouped_runs.setdefault(key, []).append(run)

    plotted_groups = []
    for key, group_runs in grouped_runs.items():
        unique_batch_sizes = {run.batch_size for run in group_runs}
        if len(unique_batch_sizes) > 1:
            plotted_groups.append(sorted(group_runs, key=lambda run: run.batch_size))

    if len(plotted_groups) == 0:
        return False

    metric_specs = [
        ("Step Time", "ms",
         lambda run: None if run.step_time_avg_sec is None else run.step_time_avg_sec * 1000.0),
        ("Throughput", "samples/s", lambda run: run.throughput_avg),
        ("Peak Allocated", "GiB", lambda run: maybe_gib(run.peak_allocated_bytes)),
        ("Peak Reserved", "GiB", lambda run: maybe_gib(run.peak_reserved_bytes)),
        ("Avg GPU Load", "%", lambda run: run.gpu_load_avg_pct),
        ("Avg GPU Mem Util", "%", lambda run: run.gpu_memory_util_avg_pct),
    ]

    all_batch_sizes = sorted({run.batch_size for group in plotted_groups for run in group})
    use_log_scale = len(all_batch_sizes) >= 3 and \
        max(all_batch_sizes) / min(all_batch_sizes) >= 4
    single_group = len(plotted_groups) == 1
    unique_models = []
    for group_runs in plotted_groups:
        model_name = group_runs[0].model
        if model_name not in unique_models:
            unique_models.append(model_name)

    figure, axes = plt.subplots(
        2, 3, figsize=(15.5, 8.2), sharex=True, constrained_layout=False)
    legend_handles = {}

    for axis_index, (axis, (title, ylabel, accessor)) in enumerate(zip(axes.flat, metric_specs)):
        for group_index, group_runs in enumerate(plotted_groups):
            x_values = [run.batch_size for run in group_runs]
            y_values = [accessor(run) for run in group_runs]
            if all(value is None for value in y_values):
                continue
            plotted_y_values = [
                math.nan if value is None else value for value in y_values]
            model_name = group_runs[0].model
            marker = MODEL_MARKERS.get(model_name, "o")
            color = MODEL_COLORS.get(model_name, "#1f77b4")
            line, = axis.plot(
                x_values,
                plotted_y_values,
                marker=marker,
                markersize=5.5,
                linewidth=2.25,
                color=color,
                label=model_name,
            )
            if model_name not in legend_handles:
                legend_handles[model_name] = line
            if single_group:
                for x_value, y_value in zip(x_values, plotted_y_values):
                    if math.isnan(y_value):
                        continue
                    label = f"{y_value:.0f}" if abs(y_value) >= 100 else f"{y_value:.2f}"
                    axis.annotate(
                        label,
                        (x_value, y_value),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=color,
                    )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.22)
        axis.set_xticks(all_batch_sizes)
        axis.set_xticklabels([str(batch_size) for batch_size in all_batch_sizes])
        if use_log_scale:
            axis.set_xscale("log", base=2)
        if axis_index < 3:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Batch Size")

    common_datasets = sorted({group_runs[0].dataset for group_runs in plotted_groups})
    common_caches = sorted({group_runs[0].cache for group_runs in plotted_groups})
    common_world_sizes = sorted({group_runs[0].world_size for group_runs in plotted_groups})
    common_edge_ratios = sorted({group_runs[0].edge_cache_ratio for group_runs in plotted_groups})
    common_node_ratios = sorted({group_runs[0].node_cache_ratio for group_runs in plotted_groups})
    subtitle_parts = []
    if len(common_datasets) == 1:
        subtitle_parts.append(f"dataset={common_datasets[0]}")
    if len(common_caches) == 1:
        subtitle_parts.append(f"cache={common_caches[0]}")
    if len(common_world_sizes) == 1:
        subtitle_parts.append(f"ws={common_world_sizes[0]}")
    if len(common_edge_ratios) == 1:
        subtitle_parts.append(f"edge={common_edge_ratios[0]:g}")
    if len(common_node_ratios) == 1:
        subtitle_parts.append(f"node={common_node_ratios[0]:g}")
    subtitle = fill(" | ".join(subtitle_parts), width=72)

    if not single_group:
        title_y = 0.992
        subtitle_y = 0.962
        legend_y = 0.932
        figure.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=min(4, len(legend_handles)),
            frameon=False,
        )
        if subtitle != "":
            figure.text(0.5, subtitle_y, subtitle, ha="center", va="top", fontsize=9.5)
            top_margin = 0.79
        else:
            top_margin = 0.81
    else:
        subtitle = fill(plotted_groups[0][0].batch_group_label(), width=78)
        figure.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=9.5)
        title_y = 0.992
        top_margin = 0.85

    figure.suptitle("Profiler Batch-Size Scaling", fontsize=16, y=title_y)
    figure.subplots_adjust(top=top_margin, wspace=0.22, hspace=0.22)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def write_selected_runs_csv(runs: Sequence[ProfileRun], output_path: Path):
    field_names = [
        "model",
        "dataset",
        "cache",
        "batch_size",
        "world_size",
        "edge_cache_ratio",
        "node_cache_ratio",
        "snapshot_time_window",
        "ingestion_batch_size",
        "step_time_avg_sec",
        "throughput_avg",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "gpu_load_avg_pct",
        "gpu_memory_util_avg_pct",
        "gpu_memory_used_avg_mb",
        "cpu_max_rss_bytes",
        "summary_path",
    ] + [f"train_{stage_key}_sec" for stage_key, _ in TRAIN_STAGE_ORDER] + \
        [f"setup_{stage_key}" for stage_key, _ in SETUP_STAGE_ORDER]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for run in runs:
            row = {
                "model": run.model,
                "dataset": run.dataset,
                "cache": run.cache,
                "batch_size": run.batch_size,
                "world_size": run.world_size,
                "edge_cache_ratio": run.edge_cache_ratio,
                "node_cache_ratio": run.node_cache_ratio,
                "snapshot_time_window": run.snapshot_time_window,
                "ingestion_batch_size": run.ingestion_batch_size,
                "step_time_avg_sec": run.step_time_avg_sec,
                "throughput_avg": run.throughput_avg,
                "peak_allocated_bytes": run.peak_allocated_bytes,
                "peak_reserved_bytes": run.peak_reserved_bytes,
                "gpu_load_avg_pct": run.gpu_load_avg_pct,
                "gpu_memory_util_avg_pct": run.gpu_memory_util_avg_pct,
                "gpu_memory_used_avg_mb": run.gpu_memory_used_avg_mb,
                "cpu_max_rss_bytes": run.cpu_max_rss_bytes,
                "summary_path": str(run.summary_path),
            }
            for stage_key, _ in TRAIN_STAGE_ORDER:
                row[f"train_{stage_key}_sec"] = run.train_stage_time_sec.get(stage_key)
            for stage_key, _ in SETUP_STAGE_ORDER:
                row[f"setup_{stage_key}"] = run.setup_time_sec.get(stage_key)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from profiler summaries.")
    parser.add_argument(
        "--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR,
        help="Directory containing profiler run directories (default: %(default)s)")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated plots will be written (default: %(default)s)")
    parser.add_argument(
        "--output-prefix", type=str, default=None,
        help="Optional filename prefix for generated plots")
    parser.add_argument(
        "--formats", nargs="*", default=["pdf"],
        help="Output plot formats, e.g. pdf png (default: pdf)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to include, e.g. TGN TGAT")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to include, e.g. REDDIT WIKI")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=None,
                        help="Batch sizes to include")
    parser.add_argument("--caches", nargs="*", default=None,
                        help="Caches to include, e.g. LRUCache LFUCache")
    parser.add_argument("--world-sizes", nargs="*", type=int, default=None,
                        help="World sizes to include")
    parser.add_argument("--edge-cache-ratios", nargs="*", type=float, default=None,
                        help="Edge cache ratios to include")
    parser.add_argument("--node-cache-ratios", nargs="*", type=float, default=None,
                        help="Node cache ratios to include")
    parser.add_argument("--snapshot-time-windows", nargs="*", type=float, default=None,
                        help="Snapshot time windows to include")
    parser.add_argument(
        "--all-matching-runs", action="store_true",
        help="Include all matching runs instead of only the latest run per config")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_paths = discover_summary_paths(args.profiles_dir)
    if len(summary_paths) == 0:
        raise SystemExit(f"No profiler summaries found under {args.profiles_dir}")

    runs = [build_profile_run(summary_path) for summary_path in summary_paths]
    runs = filter_runs(runs, args)
    if not args.all_matching_runs:
        runs = latest_per_config(runs)
    else:
        runs = sorted(runs, key=sort_runs_key)

    if len(runs) == 0:
        raise SystemExit("No profiler runs matched the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = build_output_prefix(args, runs)
    formats = [fmt.lower().lstrip(".") for fmt in args.formats]

    write_selected_runs_csv(
        runs, args.output_dir / f"{output_prefix}_selected-runs.csv")
    batch_scaling_created = False
    for plot_format in formats:
        plot_overview_dashboard(
            runs, args.output_dir / f"{output_prefix}_overview.{plot_format}")
        plot_stacked_breakdown(
            runs,
            args.output_dir / f"{output_prefix}_train-stage-breakdown.{plot_format}",
            TRAIN_STAGE_ORDER,
            TRAIN_STAGE_COLORS,
            "Training Stage Breakdown",
            lambda run, stage_key: None if run.train_stage_time_sec.get(stage_key) is None
            else run.train_stage_time_sec[stage_key] * 1000.0,
            "Time (ms)",
            normalize=False,
        )
        plot_stacked_breakdown(
            runs,
            args.output_dir / f"{output_prefix}_train-stage-share.{plot_format}",
            TRAIN_STAGE_ORDER,
            TRAIN_STAGE_COLORS,
            "Training Stage Share",
            lambda run, stage_key: run.train_stage_time_sec.get(stage_key),
            "Share of step time (%)",
            normalize=True,
        )
        plot_stacked_breakdown(
            runs,
            args.output_dir / f"{output_prefix}_setup-breakdown.{plot_format}",
            SETUP_STAGE_ORDER,
            SETUP_STAGE_COLORS,
            "Setup Stage Breakdown",
            lambda run, stage_key: run.setup_time_sec.get(stage_key),
            "Time (s)",
            normalize=False,
        )
        batch_scaling_created = plot_batch_scaling(
            runs, args.output_dir / f"{output_prefix}_batch-scaling.{plot_format}") \
            or batch_scaling_created

    print(f"Loaded {len(runs)} profiler runs.")
    print(f"Plots written to {args.output_dir}")
    print(f"Filename prefix: {output_prefix}")
    if not batch_scaling_created:
        print("Skipped batch-scaling plot because the selection did not contain multiple batch sizes per config group.")


if __name__ == "__main__":
    main()

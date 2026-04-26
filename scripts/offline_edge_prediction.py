import argparse
import contextlib
import json
import logging
import math
import os
import random
import statistics
import threading
import time
from typing import Any, Dict, List, Optional

import GPUtil
import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm

try:
    from torch.profiler import (ProfilerActivity, profile as torch_profile,
                                record_function,
                                schedule as profiler_schedule,
                                tensorboard_trace_handler)
except ImportError:
    ProfilerActivity = None
    torch_profile = None
    profiler_schedule = None
    tensorboard_trace_handler = None

    def record_function(_name):
        return contextlib.nullcontext()

import gnnflow.cache as caches
from gnnflow.config import get_default_config
from gnnflow.data import (DistributedBatchSampler, EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (DstRandEdgeSampler, EarlyStopMonitor,
                           build_dynamic_graph, get_pinned_buffers,
                           get_project_root_dir, load_dataset, load_feat,
                           mfgs_to_cuda)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE', 'GAT']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, required=True,
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--epoch", help="maximum training epoch",
                    type=int, default=50)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=8)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=8)
parser.add_argument("--print-freq", help="print frequency",
                    type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ingestion-batch-size", type=int, default=1000,
                    help="ingestion batch size")
parser.add_argument("--batch-size", type=int, default=None,
                    help="override the model default batch size")

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--edge-cache-ratio", type=float, default=0,
                    help="cache ratio for edge feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0,
                    help="cache ratio for node feature cache")
parser.add_argument("--snapshot-time-window", type=float, default=0,
                    help="time window for sampling")
parser.add_argument("--profile", action="store_true",
                    help="enable torch profiler during training")
parser.add_argument("--profile-only", action="store_true",
                    help="run only enough training steps to collect profiling data")
parser.add_argument("--profile-dir", type=str, default=None,
                    help="directory for profiler traces and summaries")
parser.add_argument("--profile-wait", type=int, default=1,
                    help="number of initial steps to skip before profiler warmup")
parser.add_argument("--profile-warmup", type=int, default=1,
                    help="number of profiler warmup steps")
parser.add_argument("--profile-active", type=int, default=6,
                    help="number of active profiler steps per cycle")
parser.add_argument("--profile-repeat", type=int, default=1,
                    help="number of profiler schedule cycles to run")
parser.add_argument("--profile-row-limit", type=int, default=50,
                    help="row limit for profiler summary tables")
parser.add_argument("--profile-gpu-sample-interval", type=float, default=0.2,
                    help="seconds between GPU utilization samples during profiling")
parser.add_argument("--profile-with-stack", action="store_true",
                    help="record stack traces in torch profiler output")
parser.add_argument("--profile-with-flops", action="store_true",
                    help="record FLOPs estimates in torch profiler output")
parser.add_argument("--profile-export-memory-timeline", action="store_true",
                    help="export a memory timeline if supported by the installed torch profiler")
parser.add_argument("--profile-record-shapes", dest="profile_record_shapes",
                    action="store_true",
                    help="record operator input shapes in torch profiler output")
parser.add_argument("--no-profile-record-shapes", dest="profile_record_shapes",
                    action="store_false",
                    help="disable operator shape recording in torch profiler output")
parser.set_defaults(profile_record_shapes=True)

args = parser.parse_args()

if args.profile_only:
    args.profile = True

if args.profile:
    if torch_profile is None:
        raise RuntimeError("torch.profiler is not available in the current torch installation")
    if args.profile_wait < 0 or args.profile_warmup < 0 or \
            args.profile_active <= 0 or args.profile_repeat <= 0:
        raise ValueError("Invalid profiler schedule values")

logging.basicConfig(level=logging.DEBUG)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

training = True


def gpu_load():
    global training
    time.sleep(5)
    while True:
        # stop when training is done
        # use a global variable to stop the thread
        if not training:
            break
        gpus = GPUtil.getGPUs()
        avg_load = sum([gpu.load for gpu in gpus]) / len(gpus)
        logging.info("GPU load: {:.2f}%".format(avg_load * 100))
        time.sleep(1)


def percentile(values: List[float], value: float) -> Optional[float]:
    if len(values) == 0:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * value
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def get_max_rss_bytes() -> Optional[int]:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system_name = os.uname().sysname.lower() if hasattr(os, "uname") else ""
    if system_name == "darwin":
        return int(usage)
    return int(usage) * 1024


def sanitize_run_component(value: Any) -> str:
    return ''.join(
        character if character.isalnum() or character in ('-', '_', '.')
        else '-'
        for character in str(value))


def maybe_synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GpuStatsMonitor(threading.Thread):
    def __init__(self, cuda_device_index: int, sample_interval: float):
        super().__init__(daemon=True)
        self._sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._physical_gpu_index = self._resolve_physical_gpu_index(
            cuda_device_index)
        self.samples: List[Dict[str, float]] = []

    @staticmethod
    def _resolve_physical_gpu_index(cuda_device_index: int) -> int:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is None:
            return cuda_device_index

        mapped_devices = []
        for token in visible_devices.split(','):
            token = token.strip()
            if token == "" or token.startswith("GPU-"):
                return cuda_device_index
            try:
                mapped_devices.append(int(token))
            except ValueError:
                return cuda_device_index

        if 0 <= cuda_device_index < len(mapped_devices):
            return mapped_devices[cuda_device_index]
        return cuda_device_index

    def run(self):
        while not self._stop_event.is_set():
            try:
                gpu = next(
                    (gpu for gpu in GPUtil.getGPUs()
                     if gpu.id == self._physical_gpu_index),
                    None)
                if gpu is not None:
                    self.samples.append({
                        "timestamp": time.time(),
                        "load_pct": float(gpu.load) * 100.0,
                        "memory_util_pct": float(gpu.memoryUtil) * 100.0,
                        "memory_used_mb": float(gpu.memoryUsed),
                        "memory_total_mb": float(gpu.memoryTotal),
                    })
            except Exception:
                pass
            self._stop_event.wait(self._sample_interval)

    def stop(self):
        self._stop_event.set()

    def summarize(self) -> Dict[str, Optional[float]]:
        if len(self.samples) == 0:
            return {
                "sample_count": 0,
                "avg_load_pct": None,
                "max_load_pct": None,
                "avg_memory_util_pct": None,
                "max_memory_util_pct": None,
                "avg_memory_used_mb": None,
                "max_memory_used_mb": None,
            }

        load_values = [sample["load_pct"] for sample in self.samples]
        memory_util_values = [
            sample["memory_util_pct"] for sample in self.samples]
        memory_used_values = [
            sample["memory_used_mb"] for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "avg_load_pct": statistics.mean(load_values),
            "max_load_pct": max(load_values),
            "avg_memory_util_pct": statistics.mean(memory_util_values),
            "max_memory_util_pct": max(memory_util_values),
            "avg_memory_used_mb": statistics.mean(memory_used_values),
            "max_memory_used_mb": max(memory_used_values),
        }


class TrainingProfiler:
    def __init__(self, args, device: torch.device, batch_size: int,
                 setup_metrics: Dict[str, float]):
        self.args = args
        self.device = device
        self.batch_size = batch_size
        self.setup_metrics = setup_metrics
        self.enabled = args.profile
        self.finished = False
        self.started = False
        self.total_steps_seen = 0
        self.step_records: List[Dict[str, Any]] = []
        self.profiler = None
        self.gpu_monitor = None

        if args.profile_dir is None:
            base_profile_dir = os.path.join(get_project_root_dir(), "profiles")
        else:
            base_profile_dir = os.path.abspath(args.profile_dir)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = "{}_{}_{}_bs{}_ranks{}_{}".format(
            sanitize_run_component(args.model),
            sanitize_run_component(args.data),
            sanitize_run_component(args.cache or "nocache"),
            batch_size,
            args.world_size,
            timestamp)
        self.output_dir = os.path.join(base_profile_dir, run_name)
        self.rank_dir = os.path.join(self.output_dir, "rank{}".format(args.rank))
        self.trace_dir = os.path.join(self.rank_dir, "traces")
        self.summary_path = os.path.join(self.rank_dir, "summary.json")
        self.aggregate_summary_path = os.path.join(
            self.output_dir, "summary_all_ranks.json")

        cycle_length = args.profile_wait + args.profile_warmup + args.profile_active
        self.total_profile_steps = cycle_length * args.profile_repeat if self.enabled else 0

        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)

    def _phase_for_step(self, step_index: int) -> str:
        cycle_length = self.args.profile_wait + self.args.profile_warmup + \
            self.args.profile_active
        if cycle_length <= 0 or step_index >= self.total_profile_steps:
            return "complete"
        offset = step_index % cycle_length
        if offset < self.args.profile_wait:
            return "wait"
        if offset < self.args.profile_wait + self.args.profile_warmup:
            return "warmup"
        return "active"

    def start(self):
        if not self.enabled or self.started:
            return

        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_monitor = GpuStatsMonitor(
                self.device.index if self.device.index is not None else 0,
                self.args.profile_gpu_sample_interval)
            self.gpu_monitor.start()

        self.profiler = torch_profile(
            activities=activities,
            schedule=profiler_schedule(
                wait=self.args.profile_wait,
                warmup=self.args.profile_warmup,
                active=self.args.profile_active,
                repeat=self.args.profile_repeat),
            on_trace_ready=tensorboard_trace_handler(
                self.trace_dir, worker_name="rank{}".format(self.args.rank)),
            record_shapes=self.args.profile_record_shapes,
            profile_memory=True,
            with_stack=self.args.profile_with_stack,
            with_flops=self.args.profile_with_flops)
        self.profiler.start()
        self.started = True

    def begin_step(self, num_samples: int) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.finished:
            return None

        maybe_synchronize(self.device)
        state = {
            "step_index": self.total_steps_seen,
            "phase": self._phase_for_step(self.total_steps_seen),
            "started_at": time.perf_counter(),
            "num_samples": num_samples,
        }
        if self.device.type == "cuda":
            state["cuda_memory_allocated_start_bytes"] = int(
                torch.cuda.memory_allocated(self.device))
            state["cuda_memory_reserved_start_bytes"] = int(
                torch.cuda.memory_reserved(self.device))
        return state

    def end_step(self, state: Optional[Dict[str, Any]],
                 stage_durations: Dict[str, float]):
        if not self.enabled or self.finished or state is None:
            return

        maybe_synchronize(self.device)
        wall_time = time.perf_counter() - state["started_at"]
        step_record: Dict[str, Any] = {
            "step_index": int(state["step_index"]),
            "phase": state["phase"],
            "num_samples": int(state["num_samples"]),
            "wall_time_sec": float(wall_time),
            "throughput_samples_per_sec":
                float(state["num_samples"]) / wall_time if wall_time > 0 else None,
            "cpu_max_rss_bytes": get_max_rss_bytes(),
        }
        for stage_name, duration in stage_durations.items():
            step_record["{}_sec".format(stage_name)] = float(duration)

        if self.device.type == "cuda":
            step_record["cuda_memory_allocated_end_bytes"] = int(
                torch.cuda.memory_allocated(self.device))
            step_record["cuda_memory_reserved_end_bytes"] = int(
                torch.cuda.memory_reserved(self.device))
            step_record["cuda_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device))
            step_record["cuda_max_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device))

        self.step_records.append(step_record)
        self.profiler.step()
        self.total_steps_seen += 1

        if self.total_steps_seen >= self.total_profile_steps:
            self.finish()

    def is_complete(self) -> bool:
        return self.finished

    @staticmethod
    def _summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
        numeric_values = [float(value) for value in values if value is not None]
        if len(numeric_values) == 0:
            return {
                "count": 0,
                "avg": None,
                "median": None,
                "min": None,
                "max": None,
                "p95": None,
            }
        return {
            "count": len(numeric_values),
            "avg": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "p95": percentile(numeric_values, 0.95),
        }

    @staticmethod
    def _extract_summary_value(summary: Dict[str, Any], path: List[str]):
        current: Any = summary
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    @classmethod
    def _aggregate_metric(cls, summaries: List[Dict[str, Any]],
                          path: List[str]) -> Dict[str, Optional[float]]:
        values = [
            cls._extract_summary_value(summary, path)
            for summary in summaries
        ]
        values = [float(value) for value in values if value is not None]
        if len(values) == 0:
            return {"avg": None, "min": None, "max": None}
        return {
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    def _build_summary(self) -> Dict[str, Any]:
        profiled_steps = [
            step for step in self.step_records if step["phase"] == "active"]
        if len(profiled_steps) == 0:
            profiled_steps = self.step_records

        stage_names = [
            "sampling",
            "feature_fetch",
            "memory_fetch",
            "memory_update",
            "memory_write_back",
            "model_forward",
            "loss_backward_optimizer",
        ]

        step_summary = self._summarize_values(
            [step["wall_time_sec"] for step in profiled_steps])
        throughput_summary = self._summarize_values([
            step["throughput_samples_per_sec"]
            for step in profiled_steps
            if step["throughput_samples_per_sec"] is not None
        ])
        stage_summaries = {
            stage_name: self._summarize_values([
                step["{}_sec".format(stage_name)]
                for step in profiled_steps
                if "{}_sec".format(stage_name) in step
            ])
            for stage_name in stage_names
        }

        summary: Dict[str, Any] = {
            "model": self.args.model,
            "dataset": self.args.data,
            "cache": self.args.cache,
            "edge_cache_ratio": self.args.edge_cache_ratio,
            "node_cache_ratio": self.args.node_cache_ratio,
            "snapshot_time_window": self.args.snapshot_time_window,
            "ingestion_batch_size": self.args.ingestion_batch_size,
            "rank": self.args.rank,
            "world_size": self.args.world_size,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "profile_only": self.args.profile_only,
            "profile_schedule": {
                "wait": self.args.profile_wait,
                "warmup": self.args.profile_warmup,
                "active": self.args.profile_active,
                "repeat": self.args.profile_repeat,
                "total_profile_steps": self.total_profile_steps,
            },
            "setup_metrics_sec": self.setup_metrics,
            "steps_seen": self.total_steps_seen,
            "profiled_steps": len(profiled_steps),
            "step_time_sec": step_summary,
            "throughput_samples_per_sec": throughput_summary,
            "stage_time_sec": stage_summaries,
            "cpu_memory": {
                "max_rss_bytes": max([
                    step["cpu_max_rss_bytes"]
                    for step in profiled_steps
                    if step.get("cpu_max_rss_bytes") is not None
                ], default=None)
            },
            "artifacts": {
                "rank_dir": self.rank_dir,
                "trace_dir": self.trace_dir,
            },
        }

        if self.device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            allocator_stats = torch.cuda.memory_stats(self.device)
            summary["cuda_memory"] = {
                "device_name": torch.cuda.get_device_name(self.device),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                "free_bytes_after_profile": int(free_bytes),
                "total_device_bytes": int(total_bytes),
                "alloc_retries": int(allocator_stats.get("num_alloc_retries", 0)),
                "ooms": int(allocator_stats.get("num_ooms", 0)),
                "active_peak_bytes": int(
                    allocator_stats.get("active_bytes.all.peak", 0)),
                "requested_peak_bytes": int(
                    allocator_stats.get("requested_bytes.all.peak", 0)),
            }
            summary["gpu_monitor"] = self.gpu_monitor.summarize() \
                if self.gpu_monitor is not None else {}
        else:
            summary["cuda_memory"] = {}
            summary["gpu_monitor"] = {}

        return summary

    def _write_profiler_tables(self):
        if self.profiler is None:
            return

        key_averages = self.profiler.key_averages()
        table_specs = [
            ("cpu_time_total", "key_averages_cpu_time.txt"),
            ("self_cpu_time_total", "key_averages_self_cpu_time.txt"),
            ("cuda_time_total", "key_averages_cuda_time.txt"),
            ("self_cuda_time_total", "key_averages_self_cuda_time.txt"),
            ("self_cpu_memory_usage", "key_averages_self_cpu_memory.txt"),
            ("self_cuda_memory_usage", "key_averages_self_cuda_memory.txt"),
        ]

        for sort_key, file_name in table_specs:
            try:
                table = key_averages.table(
                    sort_by=sort_key, row_limit=self.args.profile_row_limit)
            except Exception:
                continue
            with open(os.path.join(self.rank_dir, file_name), "w",
                      encoding="utf-8") as handle:
                handle.write(table)

        if self.args.profile_with_stack and \
                hasattr(self.profiler, "export_stacks"):
            for metric in ("self_cpu_time_total", "self_cuda_time_total"):
                try:
                    self.profiler.export_stacks(
                        os.path.join(self.rank_dir, "{}.txt".format(metric)),
                        metric)
                except Exception:
                    continue

        if self.args.profile_export_memory_timeline and \
                hasattr(self.profiler, "export_memory_timeline") and \
                self.device.type == "cuda":
            output_path = os.path.join(self.rank_dir, "memory_timeline.html")
            try:
                self.profiler.export_memory_timeline(
                    output_path, device=str(self.device))
            except TypeError:
                try:
                    self.profiler.export_memory_timeline(output_path)
                except Exception as exc:
                    logging.warning(
                        "Rank %d failed to export memory timeline: %s",
                        self.args.rank, exc)
            except Exception as exc:
                logging.warning(
                    "Rank %d failed to export memory timeline: %s",
                    self.args.rank, exc)

    def finish(self):
        if not self.enabled or self.finished:
            return

        self.finished = True

        if self.profiler is not None:
            self.profiler.stop()
        if self.gpu_monitor is not None:
            self.gpu_monitor.stop()
            self.gpu_monitor.join(timeout=2.0)

        self._write_profiler_tables()
        summary = self._build_summary()
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        if self.args.distributed:
            summaries: List[Optional[Dict[str, Any]]] = [None] * self.args.world_size
            torch.distributed.all_gather_object(summaries, summary)
            if self.args.rank == 0:
                rank_summaries = [item for item in summaries if item is not None]
                stage_time_summary = {}
                for stage_name in (
                        "sampling",
                        "feature_fetch",
                        "memory_fetch",
                        "memory_update",
                        "memory_write_back",
                        "model_forward",
                        "loss_backward_optimizer"):
                    stage_time_summary[stage_name] = self._aggregate_metric(
                        rank_summaries, ["stage_time_sec", stage_name, "avg"])
                setup_time_summary = {}
                for setup_name in (
                        "dataset_load_sec",
                        "graph_build_sec",
                        "graph_ingestion_sec",
                        "feature_load_sec",
                        "model_init_sec",
                        "cache_init_sec"):
                    setup_time_summary[setup_name] = self._aggregate_metric(
                        rank_summaries, ["setup_metrics_sec", setup_name])
                aggregate_summary = {
                    "model": self.args.model,
                    "dataset": self.args.data,
                    "cache": self.args.cache,
                    "edge_cache_ratio": self.args.edge_cache_ratio,
                    "node_cache_ratio": self.args.node_cache_ratio,
                    "snapshot_time_window": self.args.snapshot_time_window,
                    "ingestion_batch_size": self.args.ingestion_batch_size,
                    "batch_size": self.batch_size,
                    "world_size": self.args.world_size,
                    "output_dir": self.output_dir,
                    "setup_metrics_sec": setup_time_summary,
                    "step_time_sec": self._aggregate_metric(
                        rank_summaries, ["step_time_sec", "avg"]),
                    "stage_time_sec": stage_time_summary,
                    "throughput_samples_per_sec": self._aggregate_metric(
                        rank_summaries, ["throughput_samples_per_sec", "avg"]),
                    "peak_allocated_bytes": self._aggregate_metric(
                        rank_summaries, ["cuda_memory", "peak_allocated_bytes"]),
                    "peak_reserved_bytes": self._aggregate_metric(
                        rank_summaries, ["cuda_memory", "peak_reserved_bytes"]),
                    "gpu_load_pct": self._aggregate_metric(
                        rank_summaries, ["gpu_monitor", "avg_load_pct"]),
                    "gpu_memory_util_pct": self._aggregate_metric(
                        rank_summaries, ["gpu_monitor", "avg_memory_util_pct"]),
                    "ranks": rank_summaries,
                }
                with open(self.aggregate_summary_path, "w",
                          encoding="utf-8") as handle:
                    json.dump(aggregate_summary, handle, indent=2, sort_keys=True)
                logging.info(
                    "Profile summary written to %s (avg profiled step %.4fs, max peak allocated %.2f GiB)",
                    self.output_dir,
                    aggregate_summary["step_time_sec"]["avg"]
                    if aggregate_summary["step_time_sec"]["avg"] is not None else -1.0,
                    (aggregate_summary["peak_allocated_bytes"]["max"] or 0.0) /
                    (1024 ** 3))
        elif self.args.rank == 0:
            logging.info(
                "Profile summary written to %s (avg profiled step %.4fs, peak allocated %.2f GiB)",
                self.output_dir,
                summary["step_time_sec"]["avg"]
                if summary["step_time_sec"]["avg"] is not None else -1.0,
                (summary["cuda_memory"].get("peak_allocated_bytes", 0) or 0.0) /
                (1024 ** 3))


def evaluate(dataloader, sampler, model, criterion, cache, device):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in dataloader:
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid)

            if args.use_memory:
                b = mfgs[0][0]  # type: DGLBlock
                if args.distributed:
                    model.module.memory.prepare_input(b)
                    model.module.last_updated = model.module.memory_updater(b)
                else:
                    model.memory.prepare_input(b)
                    model.last_updated = model.memory_updater(b)

            pred_pos, pred_neg = model(mfgs)

            if args.use_memory:
                # NB: no need to do backward here
                # use one function
                if args.distributed:
                    model.module.memory.update_mem_mail(
                        **model.module.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)
                else:
                    model.memory.update_mem_mail(
                        **model.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)

            total_loss += criterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)),
                 torch.zeros(pred_neg.size(0))], dim=0)
            aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            aps.append(average_precision_score(y_true, y_pred))

        val_losses.append(float(total_loss))

    ap = float(torch.tensor(aps).mean())
    auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr


def main():
    args.distributed = int(os.environ.get('WORLD_SIZE', 0)) > 1
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        args.local_rank = args.rank = 0
        args.local_world_size = args.world_size = 1

    logging.info("rank: {}, world_size: {}".format(args.rank, args.world_size))

    model_config, data_config = get_default_config(args.model, args.data)
    if model_config["snapshot_time_window"] > 0 and args.data == "GDELT":
        model_config["snapshot_time_window"] = 25
    else:
        model_config["snapshot_time_window"] = args.snapshot_time_window
    logging.info("snapshot_time_window's value is {}".format(model_config["snapshot_time_window"]))
    args.use_memory = model_config['use_memory']
    if args.batch_size is not None:
        model_config['batch_size'] = args.batch_size

    if args.distributed:
        # graph is stored in shared memory
        data_config["mem_resource_type"] = "shared"

    setup_metrics: Dict[str, float] = {}

    dataset_load_start = time.time()
    train_data, val_data, test_data, full_data = load_dataset(args.data)
    setup_metrics["dataset_load_sec"] = time.time() - dataset_load_start
    train_rand_sampler = DstRandEdgeSampler(
        train_data['dst'].to_numpy(dtype=np.int32))
    val_rand_sampler = DstRandEdgeSampler(
        full_data['dst'].to_numpy(dtype=np.int32))
    test_rand_sampler = DstRandEdgeSampler(
        full_data['dst'].to_numpy(dtype=np.int32))

    train_ds = EdgePredictionDataset(train_data, train_rand_sampler)
    val_ds = EdgePredictionDataset(val_data, val_rand_sampler)
    test_ds = EdgePredictionDataset(test_data, test_rand_sampler)

    batch_size = model_config['batch_size']
    # NB: learning rate is scaled by the number of workers
    args.lr = args.lr * math.sqrt(args.world_size)
    logging.info("batch size: {}, lr: {}".format(batch_size, args.lr))

    if args.distributed:
        train_sampler = DistributedBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size,
            drop_last=False, rank=args.rank, world_size=args.world_size,
            num_chunks=args.num_chunks)
        val_sampler = DistributedBatchSampler(
            SequentialSampler(val_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
        test_sampler = DistributedBatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
    else:
        train_sampler = RandomStartBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(
            SequentialSampler(val_ds), batch_size=batch_size, drop_last=False)
        test_sampler = BatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=train_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, sampler=val_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, sampler=test_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)

    graph_build_start = time.time()
    dgraph = build_dynamic_graph(
        **data_config, device=args.local_rank)
    setup_metrics["graph_build_sec"] = time.time() - graph_build_start

    if args.distributed:
        torch.distributed.barrier()
    # insert in batch
    graph_ingestion_start = time.time()
    for i in tqdm(range(0, len(full_data), args.ingestion_batch_size)):
        batch = full_data[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=False)
        if args.distributed:
            torch.distributed.barrier()
    setup_metrics["graph_ingestion_sec"] = time.time() - graph_ingestion_start

    num_nodes = dgraph.max_vertex_id() + 1
    num_edges = dgraph.num_edges()
    # put the features in shared memory when using distributed training
    feature_load_start = time.time()
    node_feats, edge_feats = load_feat(
        args.data, shared_memory=args.distributed,
        local_rank=args.local_rank, local_world_size=args.local_world_size)
    setup_metrics["feature_load_sec"] = time.time() - feature_load_start

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(args.local_rank))
    logging.debug("device: {}".format(device))

    model_init_start = time.time()
    if args.model == "GRAPHSAGE":
        model = SAGE(dim_node, model_config['dim_embed'])
    elif args.model == 'GAT':
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    else:
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    model.to(device)

    sampler = TemporalSampler(dgraph, **model_config)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)

    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge)
    setup_metrics["model_init_sec"] = time.time() - model_init_start

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio, args.node_cache_ratio,
                                        num_nodes, num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        None,
                                        False)

    # only gnnlab static need to pass param
    cache_init_start = time.time()
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_data,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()
    setup_metrics["cache_init_sec"] = time.time() - cache_init_start

    logging.info("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    runtime_profiler = TrainingProfiler(
        args, device, batch_size, setup_metrics)

    best_e, profiled_only = train(train_loader, val_loader, sampler,
                                  model, optimizer, criterion, cache, device,
                                  runtime_profiler)

    if profiled_only:
        if args.rank == 0:
            logging.info("Profile-only run finished. Skipping checkpoint load and test evaluation.")
        return

    logging.info('Loading model at epoch {}...'.format(best_e))
    ckpt = torch.load(checkpoint_path)
    if args.distributed:
        model.module.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt['model'])
    if args.use_memory:
        if args.distributed:
            model.module.memory.restore(ckpt['memory'])
        else:
            model.memory.restore(ckpt['memory'])

    ap, auc = evaluate(test_loader, sampler, model,
                       criterion, cache, device)
    if args.distributed:
        metrics = torch.tensor([ap, auc], device=device)
        torch.distributed.all_reduce(metrics)
        metrics /= args.world_size
        ap, auc = metrics.tolist()
        if args.rank == 0:
            logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))

    if args.distributed:
        torch.distributed.barrier()


def train(train_loader, val_loader, sampler, model, optimizer, criterion,
          cache, device, runtime_profiler: TrainingProfiler):
    global training
    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()
    profiled_only = False

    next_data = None

    def sampling(target_nodes, ts, eid):
        nonlocal next_data
        sampling_start = time.perf_counter()
        with record_function("sampling"):
            mfgs = sampler.sample(target_nodes, ts)
        next_data = (mfgs, eid, time.perf_counter() - sampling_start)

    if args.local_rank == 0 and not args.profile:
        gpu_load_thread = threading.Thread(target=gpu_load)
        gpu_load_thread.start()

    logging.info('Start training...')
    if args.distributed:
        torch.distributed.barrier()
    runtime_profiler.start()

    for e in range(args.epoch):
        model.train()
        cache.reset()
        if e > 0:
            if args.distributed:
                model.module.reset()
            else:
                model.reset()
        total_loss = 0
        cache_edge_ratio_sum = 0
        cache_node_ratio_sum = 0
        total_sampling_time = 0
        total_feature_fetch_time = 0
        total_memory_fetch_time = 0
        total_memory_update_time = 0
        total_memory_write_back_time = 0
        total_model_train_time = 0
        total_samples = 0

        epoch_time_start = time.time()

        train_iter = iter(train_loader)
        target_nodes, ts, eid = next(train_iter)
        sampling_start = time.perf_counter()
        with record_function("sampling"):
            mfgs = sampler.sample(target_nodes, ts)
        next_data = (mfgs, eid, time.perf_counter() - sampling_start)

        sampling_thread = None

        i = 0
        while True:
            if sampling_thread is not None:
                sampling_thread.join()

            mfgs, eid, sampling_time = next_data
            num_target_nodes = len(eid) * 3

            # Sampling for next batch
            try:
                next_target_nodes, next_ts, next_eid = next(train_iter)
            except StopIteration:
                break
            sampling_thread = threading.Thread(target=sampling, args=(
                next_target_nodes, next_ts, next_eid))
            sampling_thread.start()

            step_state = runtime_profiler.begin_step(num_target_nodes)
            stage_durations = {
                "sampling": sampling_time,
                "feature_fetch": 0.0,
                "memory_fetch": 0.0,
                "memory_update": 0.0,
                "memory_write_back": 0.0,
                "model_forward": 0.0,
                "loss_backward_optimizer": 0.0,
            }
            total_sampling_time += sampling_time

            # Feature
            mfgs_to_cuda(mfgs, device)
            feature_start_time = time.time()
            with record_function("feature_fetch"):
                mfgs = cache.fetch_feature(
                    mfgs, eid)
            feature_duration = time.time() - feature_start_time
            total_feature_fetch_time += feature_duration
            stage_durations["feature_fetch"] = feature_duration

            if args.use_memory:
                b = mfgs[0][0]  # type: DGLBlock
                if args.distributed:
                    memory_fetch_start_time = time.time()
                    with record_function("memory_fetch"):
                        model.module.memory.prepare_input(b)
                    memory_fetch_duration = time.time() - memory_fetch_start_time
                    total_memory_fetch_time += memory_fetch_duration
                    stage_durations["memory_fetch"] = memory_fetch_duration

                    memory_update_start_time = time.time()
                    with record_function("memory_update"):
                        model.module.last_updated = model.module.memory_updater(b)
                    memory_update_duration = time.time() - memory_update_start_time
                    total_memory_update_time += memory_update_duration
                    stage_durations["memory_update"] = memory_update_duration
                else:
                    memory_fetch_start_time = time.time()
                    with record_function("memory_fetch"):
                        model.memory.prepare_input(b)
                    memory_fetch_duration = time.time() - memory_fetch_start_time
                    total_memory_fetch_time += memory_fetch_duration
                    stage_durations["memory_fetch"] = memory_fetch_duration

                    memory_update_start_time = time.time()
                    with record_function("memory_update"):
                        model.last_updated = model.memory_updater(b)
                    memory_update_duration = time.time() - memory_update_start_time
                    total_memory_update_time += memory_update_duration
                    stage_durations["memory_update"] = memory_update_duration

            # Train
            model_train_start_time = time.time()
            optimizer.zero_grad()
            with record_function("model_forward"):
                pred_pos, pred_neg = model(mfgs)
            model_forward_duration = time.time() - model_train_start_time
            total_model_train_time += model_forward_duration
            stage_durations["model_forward"] = model_forward_duration

            if args.use_memory:
                # NB: no need to do backward here
                with torch.no_grad():
                    # use one function
                    memory_write_back_start_time = time.time()
                    if args.distributed:
                        with record_function("memory_write_back"):
                            model.module.memory.update_mem_mail(
                                **model.module.last_updated, edge_feats=cache.target_edge_features,
                                neg_sample_ratio=1)
                    else:
                        with record_function("memory_write_back"):
                            model.memory.update_mem_mail(
                                **model.last_updated, edge_feats=cache.target_edge_features,
                                neg_sample_ratio=1)
                    memory_write_back_duration = time.time() - memory_write_back_start_time
                    total_memory_write_back_time += memory_write_back_duration
                    stage_durations["memory_write_back"] = memory_write_back_duration

            model_train_start_time = time.time()
            with record_function("loss_backward_optimizer"):
                loss = criterion(pred_pos, torch.ones_like(pred_pos))
                loss += criterion(pred_neg, torch.zeros_like(pred_neg))
                total_loss += float(loss) * num_target_nodes
                loss.backward()
                optimizer.step()
            loss_backward_optimizer_duration = time.time() - model_train_start_time
            stage_durations["loss_backward_optimizer"] = \
                loss_backward_optimizer_duration
            total_model_train_time += loss_backward_optimizer_duration

            cache_edge_ratio_sum += cache.cache_edge_ratio
            cache_node_ratio_sum += cache.cache_node_ratio
            total_samples += num_target_nodes
            i += 1
            runtime_profiler.end_step(step_state, stage_durations)

            if args.profile_only and runtime_profiler.is_complete():
                if sampling_thread is not None:
                    sampling_thread.join()
                    sampling_thread = None
                profiled_only = True
                break

            if (i+1) % args.print_freq == 0:
                if args.distributed:
                    metrics = torch.tensor([total_loss, cache_edge_ratio_sum,
                                            cache_node_ratio_sum, total_samples,
                                            total_sampling_time, total_feature_fetch_time,
                                            total_memory_fetch_time,
                                            total_memory_update_time,
                                            total_memory_write_back_time,
                                            total_model_train_time
                                            ]).to(device)
                    torch.distributed.all_reduce(metrics)
                    metrics /= args.world_size
                    total_loss, cache_edge_ratio_sum, cache_node_ratio_sum, \
                        total_samples, total_sampling_time, total_feature_fetch_time, \
                        total_memory_fetch_time, total_memory_update_time, \
                        total_memory_write_back_time, \
                        total_model_train_time = metrics.tolist()

                if args.rank == 0:
                    logging.info('Epoch {:d}/{:d} | Iter {:d}/{:d} | Throughput {:.2f} samples/s | Loss {:.4f} | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total Sampling Time {:.2f}s | Total Feature Fetching Time {:.2f}s | Total Memory Fetching Time {:.2f}s | Total Memory Update Time {:.2f}s | Total Memory Write Back Time {:.2f}s | Total Model Train Time {:.2f}s | Total Time {:.2f}s'.format(e + 1, args.epoch, i + 1, int(len(
                        train_loader)/args.world_size), total_samples * args.world_size / (time.time() - epoch_time_start), total_loss / (i + 1), cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_memory_write_back_time, total_model_train_time, time.time() - epoch_time_start))

        if profiled_only:
            break

        epoch_time = time.time() - epoch_time_start
        epoch_time_sum += epoch_time

        # Validation
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_loader, sampler, model, criterion, cache, device)

        if args.distributed:
            val_res = torch.tensor([val_ap, val_auc]).to(device)
            torch.distributed.all_reduce(val_res)
            val_res /= args.world_size
            val_ap, val_auc = val_res[0].item(), val_res[1].item()

        val_end = time.time()
        val_time = val_end - val_start

        if args.distributed:
            metrics = torch.tensor([val_ap, val_auc, cache_edge_ratio_sum,
                                    cache_node_ratio_sum, total_samples,
                                    total_sampling_time, total_feature_fetch_time,
                                    total_memory_fetch_time,
                                    total_memory_update_time,
                                    total_memory_write_back_time,
                                    total_model_train_time]).to(device)
            torch.distributed.all_reduce(metrics)
            metrics /= args.world_size
            val_ap, val_auc, cache_edge_ratio_sum, cache_node_ratio_sum, \
                total_samples, total_sampling_time, total_feature_fetch_time, \
                total_memory_fetch_time, total_memory_update_time, \
                total_memory_write_back_time, \
                total_model_train_time = metrics.tolist()

        if args.rank == 0:
            logging.info("Epoch {:d}/{:d} | Validation ap {:.4f} | Validation auc {:.4f} | Train time {:.2f} s | Validation time {:.2f} s | Train Throughput {:.2f} samples/s | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total Sampling Time {:.2f}s | Total Feature Fetching Time {:.2f}s | Total Memory Fetching Time {:.2f}s | Total Memory Update Time {:.2f}s | Total Memory Write Back Time {:.2f}s | Total Model Train Time {:.2f}s".format(

                e + 1, args.epoch, val_ap, val_auc, epoch_time, val_time, total_samples * args.world_size / epoch_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_memory_write_back_time, total_model_train_time))

        if args.rank == 0 and val_ap > best_ap:
            best_e = e + 1
            best_ap = val_ap
            if args.distributed:
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save({
                'model': model_to_save.state_dict(),
                'memory': model_to_save.memory.backup() if args.use_memory else None
            }, checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        # if early_stopper.early_stop_check(val_ap):
        #     logging.info("Early stop at epoch {}".format(e))
        #     break

    if args.rank == 0 and not profiled_only:
        logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))

    runtime_profiler.finish()

    if args.distributed:
        torch.distributed.barrier()

    if args.local_rank == 0 and not args.profile:
        training = False
        gpu_load_thread.join()

    return best_e, profiled_only


if __name__ == '__main__':
    main()

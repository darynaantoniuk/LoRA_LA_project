#!/usr/bin/env python3
"""Aggregate checkpoint evaluation artifacts into CSV summaries and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

_CACHE_DIR = Path.cwd() / ".cache"
_MPL_CACHE_DIR = _CACHE_DIR / "matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR_PATTERN = re.compile(r"^(.+)_(full|lora)_r(\d+)(?:_(.+))?$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
CSV_NULL = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint aggregation."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate outputs/all_checkpoint_evals JSON files and trainer_state.json "
            "artifacts into flat CSVs, pivot-style summaries, and comparison plots."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="outputs/all_checkpoint_evals",
        help="Directory containing per-checkpoint evaluation JSON files.",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Root outputs directory containing run folders and checkpoints.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory where CSVs and plots are written. Defaults to --eval-dir.",
    )
    parser.add_argument(
        "--best-by",
        type=str,
        choices=["accuracy", "f1", "throughput_samples_per_sec"],
        default="accuracy",
        help="Metric used to choose the best checkpoint for each run.",
    )
    parser.add_argument(
        "--plot-throughput",
        action="store_true",
        help="Generate a throughput-vs-rank line plot for one selected task.",
    )
    parser.add_argument(
        "--plot-task",
        type=str,
        default=None,
        help="Task name used for the throughput-vs-rank plot, for example: cola, mrpc, sst2.",
    )
    return parser.parse_args()


def parse_run_dir_name(run_dir_name: str) -> dict[str, Any]:
    """Parse task, mode, rank, and pretraining mode from a run directory name."""
    match = RUN_DIR_PATTERN.match(run_dir_name)
    if not match:
        return {
            "task": None,
            "mode": None,
            "rank": None,
            "pretraining_mode": None,
        }

    task, mode, rank_text, pretraining_mode = match.groups()
    return {
        "task": task,
        "mode": mode,
        "rank": int(rank_text),
        "pretraining_mode": pretraining_mode or "none",
    }


def parse_checkpoint_step(checkpoint_name: str) -> int | None:
    """Extract the integer step from a checkpoint directory name."""
    match = CHECKPOINT_PATTERN.match(checkpoint_name)
    return int(match.group(1)) if match else None


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_matching_eval_log(log_history: list[dict[str, Any]], checkpoint_step: int | None) -> dict[str, Any] | None:
    """Find the evaluation log matching a checkpoint step, or fall back to the latest eval log."""
    eval_logs = [entry for entry in log_history if "eval_accuracy" in entry or "eval_f1" in entry]
    if not eval_logs:
        return None
    if checkpoint_step is not None:
        for entry in eval_logs:
            if entry.get("step") == checkpoint_step:
                return entry
    return eval_logs[-1]


def load_trainer_eval_log(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load the trainer_state entry associated with one checkpoint directory."""
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return None

    trainer_state = load_json(trainer_state_path)
    checkpoint_step = parse_checkpoint_step(checkpoint_dir.name)
    return find_matching_eval_log(trainer_state.get("log_history", []), checkpoint_step)


def metric_or_none(record: dict[str, Any], key: str) -> float | None:
    """Convert a metric field to float when present."""
    value = record.get(key)
    if value is None:
        return None
    return float(value)


def enrich_with_trainer_log(record: dict[str, Any], trainer_log: dict[str, Any] | None) -> dict[str, Any]:
    """Merge trainer-state evaluation metadata into a flat output record."""
    if trainer_log is None:
        record["epoch"] = None
        record["eval_runtime"] = None
        record["eval_samples_per_second"] = None
        record["eval_steps_per_second"] = None
        return record

    record["epoch"] = trainer_log.get("epoch")
    record["eval_runtime"] = trainer_log.get("eval_runtime")
    record["eval_samples_per_second"] = trainer_log.get("eval_samples_per_second")
    record["eval_steps_per_second"] = trainer_log.get("eval_steps_per_second")

    if record.get("accuracy") is None:
        record["accuracy"] = trainer_log.get("eval_accuracy")
    if record.get("precision") is None:
        record["precision"] = trainer_log.get("eval_precision")
    if record.get("recall") is None:
        record["recall"] = trainer_log.get("eval_recall")
    if record.get("f1") is None:
        record["f1"] = trainer_log.get("eval_f1")
    if record.get("throughput_samples_per_sec") is None:
        record["throughput_samples_per_sec"] = trainer_log.get("eval_samples_per_second")
    return record


def build_record_from_eval_json(json_path: Path) -> dict[str, Any]:
    """Build one flat record from an evaluation JSON artifact."""
    payload = load_json(json_path)
    checkpoint_dir = Path(payload["checkpoint_path"])
    run_dir = checkpoint_dir.parent
    parsed = parse_run_dir_name(run_dir.name)
    trainer_log = load_trainer_eval_log(checkpoint_dir)

    record = {
        "source": "eval_json",
        "eval_json_path": str(json_path),
        "checkpoint_path": str(checkpoint_dir),
        "run_dir": run_dir.name,
        "checkpoint_name": checkpoint_dir.name,
        "checkpoint_step": parse_checkpoint_step(checkpoint_dir.name),
        "task": payload.get("task", parsed["task"]),
        "mode": payload.get("mode", parsed["mode"]),
        "rank": payload.get("rank", parsed["rank"]),
        "pretraining_mode": parsed["pretraining_mode"],
        "accuracy": metric_or_none(payload, "accuracy"),
        "precision": metric_or_none(payload, "precision"),
        "recall": metric_or_none(payload, "recall"),
        "f1": metric_or_none(payload, "f1"),
        "trainable_params": payload.get("trainable_params"),
        "trainable_percent": payload.get("trainable_percent"),
        "gpu_memory_mb": payload.get("gpu_memory_mb"),
        "throughput_samples_per_sec": payload.get("throughput_samples_per_sec"),
    }
    return enrich_with_trainer_log(record, trainer_log)


def build_record_from_trainer_state(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Build one flat record directly from trainer_state.json when no eval JSON exists."""
    run_dir = checkpoint_dir.parent
    parsed = parse_run_dir_name(run_dir.name)
    trainer_log = load_trainer_eval_log(checkpoint_dir)
    if trainer_log is None:
        return None

    return enrich_with_trainer_log(
        {
            "source": "trainer_state",
            "eval_json_path": None,
            "checkpoint_path": str(checkpoint_dir),
            "run_dir": run_dir.name,
            "checkpoint_name": checkpoint_dir.name,
            "checkpoint_step": parse_checkpoint_step(checkpoint_dir.name),
            "task": parsed["task"],
            "mode": parsed["mode"],
            "rank": parsed["rank"],
            "pretraining_mode": parsed["pretraining_mode"],
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "trainable_params": None,
            "trainable_percent": None,
            "gpu_memory_mb": None,
            "throughput_samples_per_sec": None,
        },
        trainer_log,
    )


def collect_records(eval_dir: Path, outputs_root: Path) -> list[dict[str, Any]]:
    """Collect records from eval JSONs and trainer states, preferring eval JSON when present."""
    records: list[dict[str, Any]] = []
    seen_checkpoint_paths: set[str] = set()

    for json_path in sorted(eval_dir.glob("*.json")):
        record = build_record_from_eval_json(json_path)
        records.append(record)
        seen_checkpoint_paths.add(str(Path(record["checkpoint_path"]).resolve()))

    for trainer_state_path in sorted(outputs_root.glob("*/checkpoint-*/trainer_state.json")):
        checkpoint_dir = trainer_state_path.parent
        checkpoint_path = str(checkpoint_dir.resolve())
        if checkpoint_path in seen_checkpoint_paths:
            continue
        record = build_record_from_trainer_state(checkpoint_dir)
        if record is not None:
            records.append(record)

    records.sort(
        key=lambda row: (
            row.get("task") or "",
            row.get("mode") or "",
            -1 if row.get("rank") is None else int(row["rank"]),
            row.get("pretraining_mode") or "",
            -1 if row.get("checkpoint_step") is None else int(row["checkpoint_step"]),
        )
    )
    return records


def csv_ready(value: Any) -> Any:
    """Normalize None values for CSV output."""
    return CSV_NULL if value is None else value


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to a CSV file."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_ready(row.get(field)) for field in fieldnames})


def choose_best_records(records: list[dict[str, Any]], best_by: str) -> list[dict[str, Any]]:
    """Choose the best checkpoint for each run directory using the selected metric."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["run_dir"]].append(row)

    best_rows: list[dict[str, Any]] = []
    for run_dir, rows in grouped.items():
        def sort_key(row: dict[str, Any]) -> tuple[float, float]:
            value = row.get(best_by)
            checkpoint_step = row.get("checkpoint_step") or -1
            return (-math.inf if value is None else float(value), float(checkpoint_step))

        best_row = max(rows, key=sort_key)
        best_row = dict(best_row)
        best_row["selected_by"] = best_by
        best_rows.append(best_row)

    best_rows.sort(
        key=lambda row: (
            row.get("task") or "",
            row.get("mode") or "",
            -1 if row.get("rank") is None else int(row["rank"]),
            row.get("pretraining_mode") or "",
        )
    )
    return best_rows


def variant_label(row: dict[str, Any]) -> str:
    """Build a compact label for pivot columns and plot legends."""
    mode = row.get("mode") or "unknown"
    if mode == "full":
        return "full"
    return f"lora-r{row.get('rank')}-{row.get('pretraining_mode')}"


def write_pivot_csv(path: Path, rows: list[dict[str, Any]], metric: str) -> None:
    """Write a task x variant pivot-style CSV for one metric."""
    tasks = sorted({row["task"] for row in rows if row.get("task") is not None})
    variants = sorted({variant_label(row) for row in rows})

    row_map: dict[tuple[str, str], dict[str, Any]] = {
        (row["task"], variant_label(row)): row for row in rows if row.get("task") is not None
    }

    fieldnames = ["task"] + variants
    pivot_rows = []
    for task in tasks:
        pivot_row = {"task": task}
        for variant in variants:
            item = row_map.get((task, variant))
            pivot_row[variant] = None if item is None else item.get(metric)
        pivot_rows.append(pivot_row)

    write_csv(path, fieldnames, pivot_rows)


def plot_grouped_metric(rows: list[dict[str, Any]], metric: str, title: str, out_path: Path) -> None:
    """Plot grouped bars of one metric by task and run variant."""
    tasks = sorted({row["task"] for row in rows if row.get("task") is not None})
    variants = sorted({variant_label(row) for row in rows})
    if not tasks or not variants:
        return

    values_by_variant = []
    for variant in variants:
        values = []
        for task in tasks:
            matching = next(
                (row for row in rows if row.get("task") == task and variant_label(row) == variant),
                None,
            )
            value = None if matching is None else matching.get(metric)
            values.append(float("nan") if value is None else float(value))
        values_by_variant.append(values)

    plt.figure(figsize=(max(8, 1.6 * len(tasks)), 5))
    width = 0.8 / max(1, len(variants))
    positions = list(range(len(tasks)))

    for index, (variant, values) in enumerate(zip(variants, values_by_variant)):
        offset = (index - (len(variants) - 1) / 2.0) * width
        shifted = [position + offset for position in positions]
        plt.bar(shifted, values, width=width, label=variant)

    plt.xticks(positions, tasks)
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_throughput_vs_rank_for_task(rows: list[dict[str, Any]], task: str, out_path: Path) -> None:
    """Plot throughput against LoRA rank for one task, with one line per LoRA pretraining mode."""
    task_rows = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("mode") == "lora"
        and row.get("rank") is not None
        and row.get("throughput_samples_per_sec") is not None
    ]
    if not task_rows:
        raise ValueError(f"No LoRA throughput rows available for task '{task}'.")

    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        rows_by_mode[str(row.get("pretraining_mode") or "unknown")].append(row)

    plt.figure(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v"]
    for index, mode_name in enumerate(sorted(rows_by_mode)):
        mode_rows = sorted(rows_by_mode[mode_name], key=lambda row: int(row["rank"]))
        x_values = [int(row["rank"]) for row in mode_rows]
        y_values = [float(row["throughput_samples_per_sec"]) for row in mode_rows]
        plt.plot(
            x_values,
            y_values,
            marker=markers[index % len(markers)],
            linewidth=2,
            markersize=6,
            label=mode_name,
        )

    plt.xlabel("LoRA rank")
    plt.ylabel("Throughput (samples/sec)")
    plt.title(f"Throughput vs Rank for {task.upper()} across LoRA modes")
    plt.xticks(sorted({int(row["rank"]) for row in task_rows}))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(title="LoRA mode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    """Aggregate evaluation artifacts, write CSV summaries, and plot comparisons."""
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(eval_dir, outputs_root)
    if not records:
        raise FileNotFoundError(
            f"No evaluation JSON or trainer_state artifacts found under {eval_dir} and {outputs_root}"
        )

    raw_fieldnames = [
        "source",
        "task",
        "mode",
        "rank",
        "pretraining_mode",
        "run_dir",
        "checkpoint_name",
        "checkpoint_step",
        "epoch",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "throughput_samples_per_sec",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "eval_runtime",
        "trainable_params",
        "trainable_percent",
        "gpu_memory_mb",
        "checkpoint_path",
        "eval_json_path",
    ]
    raw_csv = out_dir / "checkpoint_eval_rows.csv"
    write_csv(raw_csv, raw_fieldnames, records)

    best_rows = choose_best_records(records, best_by=args.best_by)
    best_csv = out_dir / f"best_checkpoint_per_run_by_{args.best_by}.csv"
    best_fieldnames = raw_fieldnames[:]
    best_fieldnames.insert(0, "selected_by")
    write_csv(best_csv, best_fieldnames, best_rows)

    for metric in ("accuracy", "f1", "throughput_samples_per_sec"):
        write_pivot_csv(out_dir / f"pivot_{metric}.csv", best_rows, metric)

    if args.plot_throughput:
        if args.plot_task is None:
            raise ValueError("--plot-task must be provided when --plot-throughput is used.")
        plot_throughput_vs_rank_for_task(
            best_rows,
            task=args.plot_task,
            out_path=out_dir / f"plot_throughput_vs_rank_{args.plot_task}.png",
        )

    print(f"Wrote raw checkpoint rows to {raw_csv}")
    print(f"Wrote best-per-run summary to {best_csv}")
    if args.plot_throughput:
        print(
            "Wrote pivot CSVs and throughput-vs-rank plot to "
            f"{out_dir} for task={args.plot_task}"
        )
    else:
        print(f"Wrote pivot CSVs to {out_dir}")


if __name__ == "__main__":
    main()

"""
Paper-style plots from checkpoint evaluation JSONs.

Reads:
  outputs/all_checkpoint_evals/*.json

Writes:
  outputs/plots/paper/*.png

Plots generated:
  - Per-task: rank vs metric for LoRA (best checkpoint per rank)
  - Per-task: metric vs trainable parameters (full vs LoRA ranks)
  - Per-task: throughput vs metric
  - Multi-task: small multiples for rank-vs-metric
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_eval_rows(eval_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(eval_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows


def pick_best_per_group(rows: list[dict], key_fn, metric: str) -> list[dict]:
    best: dict = {}
    for r in rows:
        key = key_fn(r)
        score = r.get(metric)
        if score is None:
            continue
        if key not in best or score > best[key].get(metric, float("-inf")):
            best[key] = r
    return list(best.values())


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def plot_rank_vs_metric(task_rows: list[dict], out_dir: Path, metric: str, task: str):
    lora = [r for r in task_rows if r.get("mode") == "lora" and r.get("rank") is not None]
    if not lora:
        return
    lora_best = pick_best_per_group(lora, key_fn=lambda r: r["rank"], metric=metric)
    lora_best.sort(key=lambda r: r["rank"])

    xs = [r["rank"] for r in lora_best]
    ys = [r[metric] for r in lora_best]

    plt.figure(figsize=(6.8, 3.8))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xticks(xs)
    plt.xlabel("LoRA rank r")
    plt.ylabel(metric.upper())
    plt.title(f"{task.upper()}: LoRA rank vs {metric.upper()} (best checkpoint)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{task}_rank_vs_{metric}.png")
    plt.close()


def plot_metric_vs_params(task_rows: list[dict], out_dir: Path, metric: str, task: str):
    rows = [r for r in task_rows if r.get(metric) is not None and r.get("trainable_params") is not None]
    if not rows:
        return

    best = pick_best_per_group(
        rows,
        key_fn=lambda r: (r.get("mode"), r.get("rank")),
        metric=metric,
    )

    def label(r: dict) -> str:
        if r.get("mode") == "lora":
            return f"lora r={r.get('rank')}"
        return "full"

    plt.figure(figsize=(6.8, 3.8))
    for r in best:
        x = r["trainable_params"]
        y = r[metric]
        if r.get("mode") == "full":
            plt.scatter([x], [y], s=90, marker="s", label="full")
        else:
            plt.scatter([x], [y], s=70, marker="o", label=label(r))
            plt.annotate(label(r), (x, y), fontsize=9, xytext=(4, 4), textcoords="offset points")

    plt.xscale("log")
    plt.xlabel("Trainable parameters (log scale)")
    plt.ylabel(metric.upper())
    plt.title(f"{task.upper()}: {metric.upper()} vs trainable parameters (best checkpoint)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{task}_{metric}_vs_params.png")
    plt.close()


def plot_throughput_vs_metric(task_rows: list[dict], out_dir: Path, metric: str, task: str):
    rows = [
        r
        for r in task_rows
        if r.get(metric) is not None
        and r.get("throughput_samples_per_sec") is not None
        and r.get("trainable_params") is not None
    ]
    if not rows:
        return

    best = pick_best_per_group(
        rows,
        key_fn=lambda r: (r.get("mode"), r.get("rank")),
        metric=metric,
    )

    plt.figure(figsize=(6.8, 3.8))
    for r in best:
        x = r["throughput_samples_per_sec"]
        y = r[metric]
        mode = r.get("mode")
        if mode == "full":
            plt.scatter([x], [y], s=90, marker="s", label="full")
        else:
            plt.scatter([x], [y], s=70, marker="o", label=f"lora r={r.get('rank')}")

    plt.xlabel("Throughput (samples/sec)")
    plt.ylabel(metric.upper())
    plt.title(f"{task.upper()}: throughput vs {metric.upper()} (best checkpoint)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{task}_throughput_vs_{metric}.png")
    plt.close()


def plot_multitask_rank_grid(all_rows: list[dict], out_dir: Path, metric: str, tasks: list[str]):
    n = len(tasks)
    if n == 0:
        return
    cols = 2
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(10.5, 3.8 * rows_n), squeeze=False)

    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        task_rows = [r for r in all_rows if r.get("task") == task and r.get("mode") == "lora"]
        task_rows = [r for r in task_rows if r.get("rank") is not None and r.get(metric) is not None]
        if not task_rows:
            ax.axis("off")
            continue
        best = pick_best_per_group(task_rows, key_fn=lambda r: r["rank"], metric=metric)
        best.sort(key=lambda r: r["rank"])
        xs = [r["rank"] for r in best]
        ys = [r[metric] for r in best]
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title(task.upper())
        ax.set_xlabel("rank r")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.25)

    # hide extra axes
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"LoRA rank vs {metric.upper()} (best checkpoint per rank)", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"multitask_rank_vs_{metric}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="outputs/all_checkpoint_evals")
    parser.add_argument("--out_dir", type=str, default="outputs/plots/paper")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1", "precision", "recall"])
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks to plot (default: all found).")
    args = parser.parse_args()

    _style()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(eval_dir)
    if not rows:
        raise SystemExit(f"No evaluation JSONs found in {eval_dir}")

    tasks = sorted({r.get("task") for r in rows if r.get("task")})
    if args.tasks:
        tasks = [t for t in tasks if t in set(args.tasks)]

    plot_multitask_rank_grid(rows, out_dir, metric=args.metric, tasks=tasks)
    for task in tasks:
        task_rows = [r for r in rows if r.get("task") == task]
        plot_rank_vs_metric(task_rows, out_dir, metric=args.metric, task=task)
        plot_metric_vs_params(task_rows, out_dir, metric=args.metric, task=task)
        plot_throughput_vs_metric(task_rows, out_dir, metric=args.metric, task=task)

    print(f"Saved paper plots to {out_dir}")


if __name__ == "__main__":
    main()
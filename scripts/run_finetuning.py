#!/usr/bin/env python3
"""Run full fine-tuning and LoRA fine-tuning jobs across GLUE tasks."""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASKS = ("sst2", "mrpc", "cola", "mnli")
DEFAULT_RANK_SWEEP = (4, 8, 16, 32)
DEFAULT_MODELS = ("roberta-base",)
DEFAULT_PRETRAINING_MODES = ("standard",)


def parse_args() -> argparse.Namespace:
    """
    Parse batch fine-tuning command-line arguments.
    Args:
        None: Reads arguments from the process command line.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    Algorithm:
        1. Register config, task, rank, model, logging, and control arguments.
        2. Parse and return the argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run full and LoRA fine-tuning jobs for selected GLUE tasks."
    )
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--rank-sweep", nargs="+", type=int, default=list(DEFAULT_RANK_SWEEP))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--pretraining-modes",
        nargs="+",
        choices=["none", "standard", "truncated_svd"],
        default=list(DEFAULT_PRETRAINING_MODES),
        help=(
            "LoRA initialization modes to run when mode=lora. "
            "Use 'standard' for regular LoRA and 'truncated_svd' for LoRA + SVD."
        ),
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default="outputs/finetuning_runs.jsonl",
        help="JSONL run log. Completed jobs in this file are skipped on reruns.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any job fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Do not include full fine-tuning baseline jobs.",
    )
    return parser.parse_args()


def load_completed_jobs(log_path: Path) -> set[tuple[str, str, str, int, str]]:
    """
    Load successful jobs from a JSONL run log.
    Args:
        log_path (Path): Path to the JSONL run log.
    Returns:
        set[tuple[str, str, str, int, str]]: Completed model, task, mode, rank, and
            LoRA pretraining mode tuples.
    Algorithm:
        1. Return an empty set when no log exists.
        2. Read each JSON line and keep records with ok status.
        3. Return the completed job identifiers.
    """
    if not log_path.exists():
        return set()

    completed = set()
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(
                    (
                        record["model"],
                        record["task"],
                        record["mode"],
                        record["rank"],
                        record.get("pretraining_mode", "none"),
                    )
                )
    return completed


def build_jobs(
    tasks: list[str],
    rank_sweep: tuple[int, ...],
    models: list[str],
    pretraining_modes: list[str],
    include_full: bool,
) -> list[dict[str, Any]]:
    """
    Build the fine-tuning job list.
    Args:
        tasks (list[str]): GLUE task names to train on.
        rank_sweep (tuple[int, ...]): LoRA ranks to evaluate.
        models (list[str]): Hugging Face model names to fine-tune.
        pretraining_modes (list[str]): LoRA initialization variants to train.
        include_full (bool): Whether to include full fine-tuning baseline jobs.
    Returns:
        list[dict[str, Any]]: Unique job dictionaries.
    Algorithm:
        1. Optionally add one full fine-tuning job for each model and task.
        2. Add LoRA jobs for every requested rank and pretraining mode.
        3. Deduplicate jobs while preserving order.
    """
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, str]] = set()

    def add_job(
        model_name: str,
        task: str,
        mode: str,
        rank: int,
        pretraining_mode: str,
    ) -> None:
        key = (model_name, task, mode, rank, pretraining_mode)
        if key in seen:
            return
        seen.add(key)
        jobs.append(
            {
                "model": model_name,
                "task": task,
                "mode": mode,
                "rank": rank,
                "pretraining_mode": pretraining_mode,
            }
        )

    for model_name in models:
        for task in tasks:
            if include_full:
                add_job(model_name, task, "full", 0, "none")
            for rank in rank_sweep:
                for pretraining_mode in pretraining_modes:
                    add_job(model_name, task, "lora", rank, pretraining_mode)
    return jobs


def write_temp_config(base_config: dict[str, Any], model_name: str) -> Path:
    """
    Write a temporary config with an overridden model name.
    Args:
        base_config (dict[str, Any]): Base YAML configuration values.
        model_name (str): Hugging Face model name for the current job.
    Returns:
        Path: Temporary YAML config path.
    Algorithm:
        1. Copy the base configuration.
        2. Replace the model_name field.
        3. Write and return a temporary YAML file.
    """
    config = dict(base_config)
    config["model_name"] = model_name
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    with temp_file as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return Path(temp_file.name)


def run_job(job: dict[str, Any], temp_config_path: Path) -> dict[str, Any]:
    """
    Run one fine-tuning job and summarize the result.
    Args:
        job (dict[str, Any]): Job definition containing model, task, mode, and rank.
        temp_config_path (Path): Config path to pass to the training entry point.
    Returns:
        dict[str, Any]: JSON-serializable run record.
    Algorithm:
        1. Build the src/train.py command for the job.
        2. Execute the command as a subprocess.
        3. Return timing, identity, status, and command metadata.
    """
    command = [
        sys.executable,
        "src/train.py",
        "--config",
        str(temp_config_path),
        "--task",
        job["task"],
        "--mode",
        job["mode"],
    ]
    if job["mode"] == "lora":
        command.extend(
            [
                "--rank",
                str(job["rank"]),
                "--pretraining-mode",
                job["pretraining_mode"],
            ]
        )

    started_at = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(command, check=False)
    ended_at = dt.datetime.now(dt.timezone.utc)

    return {
        "timestamp_utc": started_at.isoformat(),
        "finished_utc": ended_at.isoformat(),
        "model": job["model"],
        "task": job["task"],
        "mode": job["mode"],
        "rank": job["rank"],
        "pretraining_mode": job["pretraining_mode"],
        "status": "ok" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": " ".join(command),
    }


def make_interrupted_record(
    job: dict[str, Any],
    temp_config_path: Path,
) -> dict[str, Any]:
    """Build a JSONL record for a job interrupted by the user."""
    command = [
        sys.executable,
        "src/train.py",
        "--config",
        str(temp_config_path),
        "--task",
        job["task"],
        "--mode",
        job["mode"],
    ]
    if job["mode"] == "lora":
        command.extend(
            [
                "--rank",
                str(job["rank"]),
                "--pretraining-mode",
                job["pretraining_mode"],
            ]
        )

    interrupted_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "timestamp_utc": interrupted_at,
        "finished_utc": interrupted_at,
        "model": job["model"],
        "task": job["task"],
        "mode": job["mode"],
        "rank": job["rank"],
        "pretraining_mode": job["pretraining_mode"],
        "status": "interrupted",
        "return_code": 130,
        "command": " ".join(command),
    }


def main() -> None:
    """
    Run all pending fine-tuning jobs.
    Args:
        None: Reads configuration and job selections from CLI arguments.
    Returns:
        None: Writes run records and exits nonzero if any job fails.
    Algorithm:
        1. Load the base config and completed job log.
        2. Build the requested job list and skip already successful jobs.
        3. Run pending jobs, append JSONL records, and report failures.
    """
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        base_config = yaml.safe_load(file) or {}

    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed_jobs = load_completed_jobs(log_path)

    jobs = build_jobs(
        args.tasks,
        tuple(args.rank_sweep),
        args.models,
        args.pretraining_modes,
        include_full=not args.skip_full,
    )
    pending_jobs = [
        job
        for job in jobs
        if (
            job["model"],
            job["task"],
            job["mode"],
            job["rank"],
            job["pretraining_mode"],
        )
        not in completed_jobs
    ]

    print(f"Total planned jobs: {len(jobs)}")
    print(f"Already completed: {len(jobs) - len(pending_jobs)}")
    print(f"Pending now: {len(pending_jobs)}")
    print(f"Python executable: {sys.executable}")

    if args.dry_run:
        for job in pending_jobs:
            command = f"python src/train.py --task {job['task']} --mode {job['mode']}"
            if job["mode"] == "lora":
                command += (
                    f" --rank {job['rank']}"
                    f" --pretraining-mode {job['pretraining_mode']}"
                )
            print(f"[dry-run] model={job['model']} -> {command}")
        return

    any_failed = False
    for index, job in enumerate(pending_jobs, start=1):
        print(
            f"\n[{index}/{len(pending_jobs)}] model={job['model']} "
            f"task={job['task']} mode={job['mode']} rank={job['rank']} "
            f"pretraining={job['pretraining_mode']}"
        )
        temp_config_path = write_temp_config(base_config, job["model"])
        try:
            try:
                record = run_job(job, temp_config_path)
            except KeyboardInterrupt:
                record = make_interrupted_record(job, temp_config_path)
                with log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print("\nInterrupted by user. Logged the current job and stopping cleanly.")
                raise SystemExit(130) from None
        finally:
            temp_config_path.unlink(missing_ok=True)

        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        if record["status"] == "failed":
            any_failed = True
            print(f"FAILED (code={record['return_code']})")
            if args.stop_on_error:
                break
        else:
            print("OK")

    if any_failed:
        print("\nFinished with failures. Check the terminal output and run log for details.")
        sys.exit(1)

    print("\nAll pending jobs completed successfully.")


if __name__ == "__main__":
    main()

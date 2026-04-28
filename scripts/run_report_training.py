#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


DEFAULT_TASKS = ("sst2", "mrpc", "cola", "mnli")
DEFAULT_RANK_SWEEP = (4, 8, 16, 32)
DEFAULT_MODELS = ("roberta-base",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all training jobs required by the second interim report."
    )
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--rank-sweep", nargs="+", type=int, default=list(DEFAULT_RANK_SWEEP))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--output-log",
        type=str,
        default="outputs/report_training_runs.jsonl",
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
    return parser.parse_args()


def load_completed_jobs(log_path: Path):
    if not log_path.exists():
        return set()

    completed = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                completed.add((rec["model"], rec["task"], rec["mode"], rec["rank"]))
    return completed


def build_jobs(tasks, rank_sweep, models):
    jobs = []
    seen = set()

    def add_job(model_name, task, mode, rank):
        key = (model_name, task, mode, rank)
        if key in seen:
            return
        seen.add(key)
        jobs.append({"model": model_name, "task": task, "mode": mode, "rank": rank})

    for model_name in models:
        for task in tasks:
            add_job(model_name, task, "full", 0)
            add_job(model_name, task, "lora", 8)
            for rank in rank_sweep:
                add_job(model_name, task, "lora", rank)
    return jobs


def write_temp_config(base_cfg, model_name):
    cfg = dict(base_cfg)
    cfg["model_name"] = model_name
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    with tmp as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return Path(tmp.name)


def run_job(job, temp_cfg_path: Path):
    cmd = [
        sys.executable,
        "src/train.py",
        "--config",
        str(temp_cfg_path),
        "--task",
        job["task"],
        "--mode",
        job["mode"],
    ]
    if job["mode"] == "lora":
        cmd.extend(["--rank", str(job["rank"])])

    started_at = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(cmd)
    ended_at = dt.datetime.now(dt.timezone.utc)

    return {
        "timestamp_utc": started_at.isoformat(),
        "finished_utc": ended_at.isoformat(),
        "model": job["model"],
        "task": job["task"],
        "mode": job["mode"],
        "rank": job["rank"],
        "status": "ok" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": " ".join(cmd),
    }


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_jobs(log_path)

    jobs = build_jobs(args.tasks, tuple(args.rank_sweep), args.models)
    pending_jobs = [j for j in jobs if (j["model"], j["task"], j["mode"], j["rank"]) not in completed]

    print(f"Total planned jobs: {len(jobs)}")
    print(f"Already completed: {len(jobs) - len(pending_jobs)}")
    print(f"Pending now: {len(pending_jobs)}")

    if args.dry_run:
        for job in pending_jobs:
            command = f"python src/train.py --task {job['task']} --mode {job['mode']}"
            if job["mode"] == "lora":
                command += f" --rank {job['rank']}"
            print(f"[dry-run] model={job['model']} -> {command}")
        return

    any_failed = False
    for i, job in enumerate(pending_jobs, start=1):
        print(
            f"\n[{i}/{len(pending_jobs)}] model={job['model']} task={job['task']} "
            f"mode={job['mode']} rank={job['rank']}"
        )
        temp_cfg_path = write_temp_config(base_cfg, job["model"])
        try:
            record = run_job(job, temp_cfg_path)
        finally:
            temp_cfg_path.unlink(missing_ok=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

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

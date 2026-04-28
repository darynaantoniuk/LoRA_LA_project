#!/usr/bin/env python3
"""
Plot singular value spectra of LoRA update matrices ΔW = (alpha/r) * (B @ A)
for a given LoRA checkpoint.

Usage:
  python3 scripts/plot_lora_svd_spectrum.py \
    --config configs/train.yaml \
    --task cola \
    --checkpoint_path outputs/cola_lora_r8/checkpoint-1605
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import RobertaForSequenceClassification

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import load_config  # noqa: E402
from data import TASK_TO_NUM_LABELS  # noqa: E402
from lora import inject_lora, freeze_non_lora_params  # noqa: E402


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


def load_state_dict(checkpoint_path: str) -> dict:
    ckpt = Path(checkpoint_path)
    st_path = ckpt / "model.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file

        return load_file(str(st_path))
    bin_path = ckpt / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {checkpoint_path}")


def infer_rank_from_run_dir(checkpoint_path: str) -> int:
    run_dir = Path(checkpoint_path).parent.name
    # pattern: <task>_lora_r16
    if "_lora_r" not in run_dir:
        raise ValueError(f"Checkpoint does not look like LoRA run dir: {run_dir}")
    return int(run_dir.split("_lora_r")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/plots/paper")
    parser.add_argument("--max_layers", type=int, default=6, help="Plot at most N layers.")
    args = parser.parse_args()

    _style()
    cfg = load_config(args.config)
    cfg.task_name = args.task

    rank = infer_rank_from_run_dir(args.checkpoint_path)

    model = RobertaForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=TASK_TO_NUM_LABELS[cfg.task_name]
    )
    model = inject_lora(model, r=rank, lora_alpha=cfg.lora_alpha)
    freeze_non_lora_params(model)
    state_dict = load_state_dict(args.checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    spectra = []
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            delta = module.scale * (module.lora_B @ module.lora_A)
            s = torch.linalg.svdvals(delta).detach().cpu().numpy()
            spectra.append(s)
            layer_names.append(name)

    if not spectra:
        raise SystemExit("No LoRA modules found to analyze.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.2, 4.2))
    for i, (s, nm) in enumerate(list(zip(spectra, layer_names))[: args.max_layers]):
        plt.plot(s, label=f"{i}: {nm.split('.')[-2:]}")
    plt.yscale("log")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value (log scale)")
    plt.title(f"SVD spectrum of LoRA ΔW (task={args.task}, r={rank})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.task}_lora_r{rank}_svd_spectrum.png")
    plt.close()

    # Also plot an averaged spectrum (pad with NaNs)
    max_len = max(len(s) for s in spectra)
    arr = np.full((len(spectra), max_len), np.nan, dtype=np.float64)
    for i, s in enumerate(spectra):
        arr[i, : len(s)] = s
    mean_s = np.nanmean(arr, axis=0)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(mean_s, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Singular value index")
    plt.ylabel("Mean singular value (log scale)")
    plt.title(f"Mean SVD spectrum across LoRA layers (task={args.task}, r={rank})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.task}_lora_r{rank}_svd_mean_spectrum.png")
    plt.close()

    print(f"Saved SVD plots to {out_dir}")


if __name__ == "__main__":
    main()


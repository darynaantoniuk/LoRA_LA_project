#!/usr/bin/env python3
"""
Subspace similarity heatmap between LoRA A matrices from two checkpoints.

This mirrors the ICLR LoRA paper's "subspace similarity" heatmaps, but for RoBERTa.

We compute S = |Q1^T Q2| where Q1, Q2 are orthonormal bases of the column spaces
of A (shape r x d_in). We treat columns of A^T (d_in x r) as the basis vectors.

Usage:
  python3 scripts/plot_lora_subspace_similarity.py \
    --config configs/train.yaml \
    --task cola \
    --checkpoint_a outputs/cola_lora_r8/checkpoint-1605 \
    --checkpoint_b outputs/cola_lora_r32/checkpoint-1605 \
    --module_filter "attention.self.query"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import RobertaForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import load_config  # noqa: E402
from data import TASK_TO_NUM_LABELS  # noqa: E402
from lora import inject_lora, freeze_non_lora_params  # noqa: E402


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
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


def infer_rank(checkpoint_path: str) -> int:
    run_dir = Path(checkpoint_path).parent.name
    m = re.search(r"_lora_r(\d+)$", run_dir)
    if not m:
        raise ValueError(f"Cannot infer rank from run dir: {run_dir}")
    return int(m.group(1))


def build_lora_model(cfg, task: str, rank: int):
    model = RobertaForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=TASK_TO_NUM_LABELS[task]
    )
    model = inject_lora(model, r=rank, lora_alpha=cfg.lora_alpha)
    freeze_non_lora_params(model)
    model.eval()
    return model


def orthonormal_basis_from_A(A: torch.Tensor) -> np.ndarray:
    # A: (r, d_in). We want basis in R^{d_in} from columns of A^T: (d_in, r)
    M = A.detach().cpu().float().T.numpy()  # (d_in, r)
    # QR gives orthonormal basis for column space
    Q, _ = np.linalg.qr(M)
    return Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument(
        "--module_filter",
        type=str,
        default="attention.self.query",
        help="Substring to pick which LoRA module to compare.",
    )
    parser.add_argument("--layer_idx", type=int, default=0, help="Which matched module index to use.")
    parser.add_argument("--out_dir", type=str, default="outputs/plots/paper")
    args = parser.parse_args()

    _style()
    cfg = load_config(args.config)
    cfg.task_name = args.task

    rank_a = infer_rank(args.checkpoint_a)
    rank_b = infer_rank(args.checkpoint_b)

    model_a = build_lora_model(cfg, args.task, rank_a)
    model_b = build_lora_model(cfg, args.task, rank_b)

    model_a.load_state_dict(load_state_dict(args.checkpoint_a), strict=False)
    model_b.load_state_dict(load_state_dict(args.checkpoint_b), strict=False)

    modules_a = [(n, m) for n, m in model_a.named_modules() if hasattr(m, "lora_A") and args.module_filter in n]
    modules_b = [(n, m) for n, m in model_b.named_modules() if hasattr(m, "lora_A") and args.module_filter in n]
    if not modules_a or not modules_b:
        raise SystemExit("No matching LoRA modules found. Try a different --module_filter.")

    idx = args.layer_idx
    if idx >= len(modules_a) or idx >= len(modules_b):
        raise SystemExit(f"layer_idx out of range. Found {len(modules_a)} and {len(modules_b)} matches.")

    name_a, mod_a = modules_a[idx]
    name_b, mod_b = modules_b[idx]

    Qa = orthonormal_basis_from_A(mod_a.lora_A)  # (d_in, r_a)
    Qb = orthonormal_basis_from_A(mod_b.lora_A)  # (d_in, r_b)

    S = np.abs(Qa.T @ Qb)  # (r_a, r_b)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(S, aspect="auto", interpolation="nearest")
    plt.colorbar(label="|Q_a^T Q_b|")
    plt.xlabel(f"r={rank_b} directions")
    plt.ylabel(f"r={rank_a} directions")
    plt.title(f"Subspace similarity (A)\\n{args.task}: {args.module_filter} (idx={idx})")
    plt.tight_layout()
    fname = f"{args.task}_subspace_A_{args.module_filter.replace('.', '_')}_r{rank_a}_vs_r{rank_b}_idx{idx}.png"
    plt.savefig(out_dir / fname)
    plt.close()

    print(f"Saved heatmap to {out_dir / fname}")


if __name__ == "__main__":
    main()


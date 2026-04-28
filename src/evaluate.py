import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import precision_recall_fscore_support
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

from config import load_config
from data import load_glue_dataset, tokenize_dataset, get_eval_split, TASK_TO_NUM_LABELS
from lora import count_parameters, inject_lora, freeze_non_lora_params


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    average = "binary" if len(np.unique(labels)) <= 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_eval_trainer(model, tokenizer, eval_dataset, output_dir="outputs/eval_tmp"):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        report_to="none",
    )
    kwargs = {
        "model": model,
        "args": args,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
    }

    params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer

    return Trainer(**kwargs)


def _load_state_dict_from_checkpoint(checkpoint_path: str):
    ckpt = Path(checkpoint_path)
    safetensors_path = ckpt / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    bin_path = ckpt / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found in checkpoint: {checkpoint_path}"
    )


def _infer_mode_and_rank(checkpoint_path: str):
    run_dir_name = Path(checkpoint_path).parent.name
    match = re.match(r"^(.+)_(full|lora)_r(\d+)$", run_dir_name)
    if not match:
        return None, None
    mode = match.group(2)
    rank = int(match.group(3))
    return mode, rank


def run_checkpoint_evaluation(cfg, checkpoint_path: str):
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)
    raw_dataset = load_glue_dataset(cfg.task_name)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, cfg.task_name, cfg.max_length)
    eval_dataset = get_eval_split(tokenized, cfg.task_name)

    mode, rank = _infer_mode_and_rank(checkpoint_path)
    if mode == "lora":
        model = RobertaForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=TASK_TO_NUM_LABELS[cfg.task_name],
        )
        model = inject_lora(model, r=rank, lora_alpha=cfg.lora_alpha)
        freeze_non_lora_params(model)
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys while loading LoRA checkpoint {checkpoint_path}: "
                f"{unexpected[:8]}"
            )
        if missing:
            raise RuntimeError(
                f"Missing keys while loading LoRA checkpoint {checkpoint_path}: "
                f"{missing[:8]}"
            )
    else:
        model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)

    params = count_parameters(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = build_eval_trainer(model, tokenizer, eval_dataset)

    start = time.time()
    metrics = trainer.evaluate()
    elapsed = time.time() - start

    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    throughput = len(eval_dataset) / elapsed if elapsed > 0 else None

    result = {
        "checkpoint_path": checkpoint_path,
        "task": cfg.task_name,
        "mode": mode or "unknown",
        "rank": rank,
        "accuracy": metrics.get("eval_accuracy"),
        "precision": metrics.get("eval_precision"),
        "recall": metrics.get("eval_recall"),
        "f1": metrics.get("eval_f1"),
        "trainable_params": params["trainable"],
        "trainable_percent": params["trainable_percent"],
        "gpu_memory_mb": peak_memory_mb,
        "throughput_samples_per_sec": throughput,
    }
    return result


def run_rank_sweep(cfg, ranks=(4, 8, 16, 32)):
    results = []
    cfg.mode = "lora"
    for r in ranks:
        results.append(run_single_experiment(cfg, rank=r))
    return results


def plot_rank_sweep(results, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    ranks = [r["rank"] for r in results]
    accs = [r["accuracy"] for r in results]
    params = [r["trainable_params"] for r in results]

    plt.figure(figsize=(8, 4))
    plt.plot(ranks, accs, marker="o")
    plt.xlabel("Rank r")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs LoRA Rank")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rank_vs_accuracy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar([str(r) for r in ranks], params)
    plt.xlabel("Rank r")
    plt.ylabel("Trainable parameters")
    plt.title("Trainable Parameters vs LoRA Rank")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rank_vs_params.png"), dpi=150)
    plt.close()


def implicit_simultaneous_power_iteration(B, A, scale, k, num_iters=50, tol=1e-6):
    d_out, r = B.shape
    r_A, d_in = A.shape
    assert r == r_A, "Inner dimensions of LoRA matrices must match."
    
    V = torch.randn(d_in, k, device=B.device, dtype=B.dtype)
    V, _ = torch.linalg.qr(V)
    
    prev_s = None
    
    for _ in range(num_iters):
        AV = A @ V
        U = scale * (B @ AV)
        U, _ = torch.linalg.qr(U)
        
        BtU = B.T @ U
        V = scale * (A.T @ BtU)
        V, _ = torch.linalg.qr(V)

        s = torch.diag(U.T @ (scale * (B @ (A @ V))))
        s = torch.abs(s)
        
        if prev_s is not None and torch.max(torch.abs(s - prev_s)) < tol:
            break
        prev_s = s
        
    return torch.sort(s, descending=True).values

def svd_analysis_of_lora(model, save_dir="outputs", num_iters=50):
    os.makedirs(save_dir, exist_ok=True)

    singular_values = []
    
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                B = module.lora_B
                A = module.lora_A
                scale = module.scale
                
                r = A.shape[0] 
                
                s = implicit_simultaneous_power_iteration(B, A, scale, k=r, num_iters=num_iters)
                singular_values.append(s.cpu().numpy())

    if not singular_values:
        return

    plt.figure(figsize=(8, 4))
    for i, s in enumerate(singular_values[:5]):
        plt.plot(s, marker='o', markersize=4, label=f"Layer {i}")
        
    plt.yscale("log")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("SVD Spectrum of LoRA Updates (Orthogonal Iteration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svd_spectrum.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.checkpoint_path}"
        )

    cfg = load_config(args.config)
    if args.task is not None:
        cfg.task_name = args.task

    os.makedirs(cfg.output_dir, exist_ok=True)

    result = run_checkpoint_evaluation(cfg, args.checkpoint_path)
    output_file = args.output_file or os.path.join(cfg.output_dir, "checkpoint_eval_result.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
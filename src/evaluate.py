import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

from config import load_config
from data import load_glue_dataset, tokenize_dataset, get_eval_split, TASK_TO_NUM_LABELS
from lora import inject_lora, freeze_non_lora_params, count_parameters


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


def build_eval_trainer(model, tokenizer, eval_dataset, output_dir="outputs/eval_tmp"):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def run_single_experiment(cfg, rank=None):
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)
    raw_dataset = load_glue_dataset(cfg.task_name)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, cfg.task_name, cfg.max_length)
    eval_dataset = get_eval_split(tokenized, cfg.task_name)

    model = RobertaForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=TASK_TO_NUM_LABELS[cfg.task_name],
    )

    if cfg.mode == "lora":
        model = inject_lora(model, r=rank or cfg.rank, lora_alpha=cfg.lora_alpha)
        freeze_non_lora_params(model)

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
        "task": cfg.task_name,
        "mode": cfg.mode,
        "rank": rank if rank is not None else cfg.rank,
        "accuracy": metrics.get("eval_accuracy"),
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


def svd_analysis_of_lora(model, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    singular_values = []
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            delta = module.scale * (module.lora_B @ module.lora_A)
            s = torch.linalg.svdvals(delta).detach().cpu().numpy()
            singular_values.append(s)

    if not singular_values:
        return

    plt.figure(figsize=(8, 4))
    for i, s in enumerate(singular_values[:5]):
        plt.plot(s, label=f"Layer {i}")
    plt.yscale("log")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("SVD Spectrum of LoRA Updates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svd_spectrum.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--rank_sweep", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task is not None:
        cfg.task_name = args.task

    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.rank_sweep:
        results = run_rank_sweep(cfg)
        with open(os.path.join(cfg.output_dir, "rank_sweep_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        plot_rank_sweep(results, save_dir=cfg.output_dir)
        print(results)
    else:
        result = run_single_experiment(cfg)
        with open(os.path.join(cfg.output_dir, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(result)


if __name__ == "__main__":
    main()
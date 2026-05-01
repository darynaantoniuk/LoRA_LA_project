"""Checkpoint evaluation and LoRA spectrum analysis utilities."""

import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from config import TrainConfig, load_config
from data import TASK_TO_NUM_LABELS, get_eval_split, load_glue_dataset, tokenize_dataset
from lora import count_parameters, freeze_non_lora_params, inject_lora


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """
    Compute classification metrics from trainer predictions.
    Args:
        eval_pred (Any): Tuple-like object containing logits and labels.
    Returns:
        dict[str, float]: Accuracy, precision, recall, and F1 score.
    Algorithm:
        1. Convert logits to predicted labels with argmax.
        2. Select binary or macro averaging from the label set.
        3. Compute and return scalar metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()
    average = "binary" if len(np.unique(labels)) <= 2 else "macro"
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
    }


def build_eval_trainer(
    model: torch.nn.Module,
    tokenizer: RobertaTokenizer,
    eval_dataset: Any,
    output_dir: str = "outputs/eval_tmp",
) -> Trainer:
    """
    Build a Trainer configured for evaluation only.
    Args:
        model (torch.nn.Module): Model to evaluate.
        tokenizer (RobertaTokenizer): Tokenizer or processing class for Trainer compatibility.
        eval_dataset (Any): Evaluation dataset split.
        output_dir (str): Temporary output directory for Trainer artifacts.
    Returns:
        Trainer: Evaluation-ready Trainer instance.
    Algorithm:
        1. Create evaluation TrainingArguments.
        2. Prepare shared Trainer keyword arguments.
        3. Attach tokenizer or processing_class based on installed transformers API.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        report_to="none",
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


def load_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """
    Load a PyTorch state dictionary from a checkpoint directory.
    Args:
        checkpoint_path (str): Directory containing model.safetensors or pytorch_model.bin.
    Returns:
        dict[str, torch.Tensor]: Loaded model state dictionary.
    Algorithm:
        1. Check for a safetensors checkpoint first.
        2. Fallback to a PyTorch binary checkpoint.
        3. Raise FileNotFoundError when neither file exists.
    """
    checkpoint_dir = Path(checkpoint_path)
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    binary_path = checkpoint_dir / "pytorch_model.bin"
    if binary_path.exists():
        return torch.load(str(binary_path), map_location="cpu")

    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found in checkpoint: {checkpoint_path}"
    )


def infer_mode_and_rank(checkpoint_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Infer training mode and LoRA rank from a checkpoint parent directory name.
    Args:
        checkpoint_path (str): Path to a checkpoint directory.
    Returns:
        Tuple[Optional[str], Optional[int]]: Inferred mode and rank, or None values.
    Algorithm:
        1. Read the parent run directory name.
        2. Match the expected task_mode_rRank pattern.
        3. Return parsed mode and rank when available.
    """
    run_dir_name = Path(checkpoint_path).parent.name
    match = re.match(r"^(.+)_(full|lora)_r(\d+)(?:_.+)?$", run_dir_name)
    if not match:
        return None, None
    return match.group(2), int(match.group(3))


def load_checkpoint_model(
    config: TrainConfig,
    checkpoint_path: str,
    mode: Optional[str],
    rank: Optional[int],
) -> torch.nn.Module:
    """
    Load a full or LoRA checkpoint model for evaluation.
    Args:
        config (TrainConfig): Evaluation configuration.
        checkpoint_path (str): Checkpoint directory path.
        mode (Optional[str]): Inferred training mode.
        rank (Optional[int]): Inferred LoRA rank.
    Returns:
        torch.nn.Module: Model with checkpoint weights loaded.
    Algorithm:
        1. Load a full checkpoint directly unless LoRA mode is inferred.
        2. Rebuild the base model and inject LoRA adapters for LoRA checkpoints.
        3. Load the checkpoint state dict with strict key validation.
    """
    if mode != "lora":
        return RobertaForSequenceClassification.from_pretrained(checkpoint_path)

    if rank is None:
        raise ValueError(f"Could not infer LoRA rank from checkpoint: {checkpoint_path}")

    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=TASK_TO_NUM_LABELS[config.task_name],
    )
    model = inject_lora(model, r=rank, lora_alpha=config.lora_alpha)
    freeze_non_lora_params(model)

    state_dict = load_state_dict_from_checkpoint(checkpoint_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys while loading LoRA checkpoint {checkpoint_path}: "
            f"{unexpected_keys[:8]}"
        )
    if missing_keys:
        raise RuntimeError(
            f"Missing keys while loading LoRA checkpoint {checkpoint_path}: {missing_keys[:8]}"
        )
    return model


def run_checkpoint_evaluation(config: TrainConfig, checkpoint_path: str) -> dict[str, Any]:
    """
    Evaluate one checkpoint and collect metrics, parameter counts, and speed data.
    Args:
        config (TrainConfig): Evaluation configuration.
        checkpoint_path (str): Checkpoint directory path.
    Returns:
        dict[str, Any]: Evaluation result summary.
    Algorithm:
        1. Load tokenizer, dataset, and evaluation split.
        2. Infer checkpoint mode and load the matching model.
        3. Run Trainer evaluation and collect runtime statistics.
    """
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)
    raw_dataset = load_glue_dataset(config.task_name)
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config.task_name,
        config.max_length,
    )
    eval_dataset = get_eval_split(tokenized_dataset, config.task_name)

    mode, rank = infer_mode_and_rank(checkpoint_path)
    model = load_checkpoint_model(config, checkpoint_path, mode, rank)
    parameter_stats = count_parameters(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = build_eval_trainer(model, tokenizer, eval_dataset)
    start_time = time.time()
    metrics = trainer.evaluate()
    elapsed_seconds = time.time() - start_time

    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    throughput = len(eval_dataset) / elapsed_seconds if elapsed_seconds > 0 else None
    return {
        "checkpoint_path": checkpoint_path,
        "task": config.task_name,
        "mode": mode or "unknown",
        "rank": rank,
        "accuracy": metrics.get("eval_accuracy"),
        "precision": metrics.get("eval_precision"),
        "recall": metrics.get("eval_recall"),
        "f1": metrics.get("eval_f1"),
        "trainable_params": parameter_stats["trainable"],
        "trainable_percent": parameter_stats["trainable_percent"],
        "gpu_memory_mb": peak_memory_mb,
        "throughput_samples_per_sec": throughput,
    }


def implicit_simultaneous_power_iteration(
    lora_b: torch.Tensor,
    lora_a: torch.Tensor,
    scale: float,
    rank: int,
    num_iters: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate singular values of an implicit LoRA update matrix.
    Args:
        lora_b (torch.Tensor): LoRA B matrix with shape [out_features, rank].
        lora_a (torch.Tensor): LoRA A matrix with shape [rank, in_features].
        scale (float): LoRA scaling factor.
        rank (int): Number of singular values to estimate.
        num_iters (int): Maximum orthogonal iterations.
        tol (float): Convergence tolerance for singular values.
    Returns:
        torch.Tensor: Estimated singular values sorted descending.
    Algorithm:
        1. Initialize an orthonormal right subspace.
        2. Alternately project through scale * B A and its transpose.
        3. Stop when estimated singular values stabilize.
    """
    out_rank = lora_b.shape[1]
    in_rank = lora_a.shape[0]
    if out_rank != in_rank:
        raise ValueError("Inner dimensions of LoRA matrices must match.")

    right_vectors = torch.randn(
        lora_a.shape[1],
        rank,
        device=lora_b.device,
        dtype=lora_b.dtype,
    )
    right_vectors, _ = torch.linalg.qr(right_vectors)
    previous_values = None

    for _ in range(num_iters):
        left_vectors = scale * (lora_b @ (lora_a @ right_vectors))
        left_vectors, _ = torch.linalg.qr(left_vectors)
        right_vectors = scale * (lora_a.T @ (lora_b.T @ left_vectors))
        right_vectors, _ = torch.linalg.qr(right_vectors)

        values = torch.abs(
            torch.diag(left_vectors.T @ (scale * (lora_b @ (lora_a @ right_vectors))))
        )
        if previous_values is not None and torch.max(torch.abs(values - previous_values)) < tol:
            break
        previous_values = values

    return torch.sort(values, descending=True).values


def svd_analysis_of_lora(
    model: torch.nn.Module,
    save_dir: str = "outputs",
    num_iters: int = 50,
) -> None:
    """
    Plot estimated singular-value spectra for LoRA update matrices.
    Args:
        model (torch.nn.Module): Model containing optional LoRA modules.
        save_dir (str): Directory where the plot is saved.
        num_iters (int): Orthogonal iterations used for each LoRA update.
    Returns:
        None: Saves a spectrum image when LoRA modules exist.
    Algorithm:
        1. Find modules containing lora_A and lora_B matrices.
        2. Estimate singular values of each implicit LoRA update.
        3. Plot the first few spectra and save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    singular_value_sets = []

    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                singular_values = implicit_simultaneous_power_iteration(
                    module.lora_B,
                    module.lora_A,
                    module.scale,
                    rank=module.lora_A.shape[0],
                    num_iters=num_iters,
                )
                singular_value_sets.append(singular_values.cpu().numpy())

    if not singular_value_sets:
        return

    plt.figure(figsize=(8, 4))
    for layer_index, singular_values in enumerate(singular_value_sets[:5]):
        plt.plot(singular_values, marker="o", markersize=4, label=f"Layer {layer_index}")

    plt.yscale("log")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("SVD Spectrum of LoRA Updates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svd_spectrum.png"), dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    """
    Parse checkpoint-evaluation command-line arguments.
    Args:
        None: Reads arguments from the process command line.
    Returns:
        argparse.Namespace: Parsed CLI argument namespace.
    Algorithm:
        1. Create the argument parser.
        2. Register config, task, checkpoint, and output-file arguments.
        3. Parse and return command-line values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """
    Evaluate a checkpoint and save the result as JSON.
    Args:
        None: Reads configuration and checkpoint arguments from CLI.
    Returns:
        None: Writes evaluation metrics to disk and prints them.
    Algorithm:
        1. Parse arguments and validate checkpoint directory.
        2. Load configuration and apply optional task override.
        3. Run evaluation and write the JSON result file.
    """
    args = parse_args()
    if not os.path.isdir(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_path}")

    config = load_config(args.config)
    if args.task is not None:
        config.task_name = args.task

    os.makedirs(config.output_dir, exist_ok=True)
    result = run_checkpoint_evaluation(config, args.checkpoint_path)
    output_file = args.output_file or os.path.join(
        config.output_dir,
        "checkpoint_eval_result.json",
    )
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(result)


if __name__ == "__main__":
    main()

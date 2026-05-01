"""Model training entry point for full fine-tuning and LoRA fine-tuning."""

import argparse
import inspect
import os
import random
from typing import Any, Optional

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


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training runs.
    Args:
        seed (int): Seed value applied to Python, NumPy, and PyTorch.
    Returns:
        None: Updates global random states in place.
    Algorithm:
        1. Seed Python's random module.
        2. Seed NumPy and PyTorch CPU generators.
        3. Seed CUDA generators when CUDA is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """
    Compute classification metrics from trainer predictions.
    Args:
        eval_pred (Any): Tuple-like object containing logits and labels.
    Returns:
        dict[str, float]: Accuracy, precision, recall, and F1 score.
    Algorithm:
        1. Convert logits to predicted labels with argmax.
        2. Choose binary or macro averaging from label cardinality.
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


def build_model(config: TrainConfig) -> RobertaForSequenceClassification:
    """
    Build a sequence-classification model with optional LoRA adapters.
    Args:
        config (TrainConfig): Training and model configuration.
    Returns:
        RobertaForSequenceClassification: Initialized model ready for training.
    Algorithm:
        1. Load the pretrained classifier with the task label count.
        2. Inject LoRA adapters when LoRA mode is selected.
        3. Freeze non-LoRA parameters for adapter-only training.
    """
    num_labels = TASK_TO_NUM_LABELS[config.task_name]
    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
    )

    if config.mode == "lora":
        model = inject_lora(
            model,
            r=config.rank,
            lora_alpha=config.lora_alpha,
            pretraining_mode=config.pretraining_mode,
            svd_rank_ratio=config.svd_rank_ratio,
            svd_energy_threshold=config.svd_energy_threshold,
            svd_max_iter=config.svd_max_iter,
            svd_tol=config.svd_tol,
            svd_seed=config.seed,
        )
        freeze_non_lora_params(model)

    return model


def make_output_dir(config: TrainConfig) -> str:
    """
    Build the output directory name for a training run.
    Args:
        config (TrainConfig): Training configuration used to name the run.
    Returns:
        str: Output directory path for checkpoints and logs.
    Algorithm:
        1. Combine root output directory, task, mode, rank, and pretraining mode.
        2. Return the formatted path string.
    """
    run_name = f"{config.task_name}_{config.mode}_r{config.rank}_{config.pretraining_mode}"
    return os.path.join(config.output_dir, run_name)


def make_training_args(config: TrainConfig) -> TrainingArguments:
    """
    Create Hugging Face TrainingArguments with version-compatible options.
    Args:
        config (TrainConfig): Training hyperparameter configuration.
    Returns:
        TrainingArguments: Configured Trainer arguments.
    Algorithm:
        1. Prepare common training, evaluation, logging, and saving options.
        2. Detect whether the installed transformers version expects evaluation_strategy or eval_strategy.
        3. Construct and return TrainingArguments.
    """
    training_kwargs = {
        "output_dir": make_output_dir(config),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": False,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": config.seed,
    }

    argument_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in argument_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in argument_params:
        training_kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**training_kwargs)


def make_trainer(
    model: RobertaForSequenceClassification,
    training_args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: RobertaTokenizer,
) -> Trainer:
    """
    Create a Hugging Face Trainer with tokenizer API compatibility.
    Args:
        model (RobertaForSequenceClassification): Model to train and evaluate.
        training_args (TrainingArguments): Trainer runtime arguments.
        train_dataset (Any): Training dataset split.
        eval_dataset (Any): Evaluation dataset split.
        tokenizer (RobertaTokenizer): Tokenizer or processing class.
    Returns:
        Trainer: Configured trainer instance.
    Algorithm:
        1. Prepare the shared Trainer keyword arguments.
        2. Detect whether Trainer expects tokenizer or processing_class.
        3. Construct and return the Trainer.
    """
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


def apply_cli_overrides(
    config: TrainConfig,
    task_name: Optional[str],
    mode: Optional[str],
    rank: Optional[int],
    pretraining_mode: Optional[str],
) -> TrainConfig:
    """
    Apply command-line overrides to a loaded configuration object.
    Args:
        config (TrainConfig): Configuration to mutate.
        task_name (Optional[str]): Optional GLUE task override.
        mode (Optional[str]): Optional training mode override.
        rank (Optional[int]): Optional LoRA rank override.
        pretraining_mode (Optional[str]): Optional pretraining mode override.
    Returns:
        TrainConfig: Mutated configuration object.
    Algorithm:
        1. Check each optional CLI value.
        2. Replace the matching config field when provided.
        3. Return the updated config.
    """
    if task_name is not None:
        config.task_name = task_name
    if mode is not None:
        config.mode = mode
    if rank is not None:
        config.rank = rank
    if pretraining_mode is not None:
        config.pretraining_mode = pretraining_mode
    return config


def parse_args() -> argparse.Namespace:
    """
    Parse training command-line arguments.
    Args:
        None: Reads arguments from the process command line.
    Returns:
        argparse.Namespace: Parsed CLI argument namespace.
    Algorithm:
        1. Create the argument parser.
        2. Register config, task, mode, rank, and pretraining overrides.
        3. Parse and return command-line values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["full", "lora"], default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument(
        "--pretraining-mode",
        type=str,
        choices=["none", "standard", "truncated_svd"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    """
    Run model training and final evaluation.
    Args:
        None: Reads configuration from YAML and command-line arguments.
    Returns:
        None: Trains the model and prints final metrics.
    Algorithm:
        1. Load config and apply CLI overrides.
        2. Set seeds, tokenizer, datasets, and model.
        3. Build Trainer, run training, and print evaluation metrics.
    """
    args = parse_args()
    config = apply_cli_overrides(
        load_config(args.config),
        task_name=args.task,
        mode=args.mode,
        rank=args.rank,
        pretraining_mode=args.pretraining_mode,
    )

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    raw_dataset = load_glue_dataset(config.task_name)
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config.task_name,
        config.max_length,
    )

    model = build_model(config)
    print("Parameter statistics:", count_parameters(model))

    trainer = make_trainer(
        model=model,
        training_args=make_training_args(config),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=get_eval_split(tokenized_dataset, config.task_name),
        tokenizer=tokenizer,
    )
    trainer.train()
    print("Final evaluation metrics:", trainer.evaluate())


if __name__ == "__main__":
    main()

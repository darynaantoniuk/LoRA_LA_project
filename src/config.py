"""Configuration loading for the training pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainConfig:
    """Training, evaluation, LoRA, and SVD pretraining configuration."""

    model_name: str = "roberta-base"
    task_name: str = "sst2"
    mode: str = "lora"
    rank: int = 8
    lora_alpha: int = 16

    pretraining_mode: str = "standard"
    svd_rank_ratio: Optional[float] = None
    svd_energy_threshold: Optional[float] = None
    svd_max_iter: int = 100
    svd_tol: float = 1e-6

    learning_rate: float = 2e-4
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    max_length: int = 128
    output_dir: str = "outputs"
    seed: int = 42


def load_config(path: str = "configs/train.yaml") -> TrainConfig:
    """
    Load training configuration from a YAML file or return defaults.
    Args:
        path (str): Path to the YAML configuration file.
    Returns:
        TrainConfig: Parsed configuration object.
    Algorithm:
        1. Resolve the configuration file path.
        2. Return default values when the file does not exist.
        3. Load YAML values and unpack them into TrainConfig.
    """
    config_path = Path(path)
    if not config_path.exists():
        return TrainConfig()

    with config_path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file) or {}

    return TrainConfig(**config_data)

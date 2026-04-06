from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    model_name: str = "roberta-base"
    task_name: str = "sst2"
    mode: str = "lora"  # "lora" or "full"
    rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    max_length: int = 128
    output_dir: str = "outputs"
    seed: int = 42


def load_config(path: str = "configs/train.yaml") -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        return TrainConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return TrainConfig(**data)
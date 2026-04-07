import argparse
import os
import random
import numpy as np
import torch

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from config import load_config
from data import load_glue_dataset, tokenize_dataset, get_eval_split, TASK_TO_NUM_LABELS
from lora import inject_lora, freeze_non_lora_params, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


def build_model(model_name: str, task_name: str, mode: str, rank: int, lora_alpha: int):
    num_labels = TASK_TO_NUM_LABELS[task_name]
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if mode == "lora":
        model = inject_lora(model, r=rank, lora_alpha=lora_alpha)
        freeze_non_lora_params(model)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["full", "lora"], default=None)
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.task is not None:
        cfg.task_name = args.task
    if args.mode is not None:
        cfg.mode = args.mode
    if args.rank is not None:
        cfg.rank = args.rank

    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)
    raw_dataset = load_glue_dataset(cfg.task_name)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, cfg.task_name, cfg.max_length)

    train_dataset = tokenized["train"]
    eval_dataset = get_eval_split(tokenized, cfg.task_name)

    model = build_model(
        model_name=cfg.model_name,
        task_name=cfg.task_name,
        mode=cfg.mode,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
    )

    stats = count_parameters(model)
    print("Parameter statistics:", stats)

    training_args = TrainingArguments(
        output_dir=f"{cfg.output_dir}/{cfg.task_name}_{cfg.mode}_r{cfg.rank}",
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    print("Final evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
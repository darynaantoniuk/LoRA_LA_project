"""Dataset loading and tokenization utilities."""

from typing import Any, Optional, Tuple

from datasets import DatasetDict, load_dataset


TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
}

TASK_TO_NUM_LABELS = {
    "sst2": 2,
    "mrpc": 2,
    "cola": 2,
    "mnli": 3,
}


def load_glue_dataset(task_name: str) -> DatasetDict:
    """
    Load a GLUE dataset by task name.
    Args:
        task_name (str): GLUE task identifier.
    Returns:
        DatasetDict: Raw train, validation, and test splits.
    Algorithm:
        1. Pass the GLUE namespace and task name to Hugging Face datasets.
        2. Return the loaded dataset dictionary.
    """
    return load_dataset("glue", task_name)


def get_task_text_keys(task_name: str) -> Tuple[str, Optional[str]]:
    """
    Return the text column names used by a GLUE task.
    Args:
        task_name (str): GLUE task identifier.
    Returns:
        Tuple[str, Optional[str]]: Primary and optional secondary text keys.
    Algorithm:
        1. Validate that the task name is supported.
        2. Read the matching text-column pair.
        3. Return the pair to the caller.
    """
    if task_name not in TASK_TO_KEYS:
        raise KeyError(f"Unsupported GLUE task: {task_name}")
    return TASK_TO_KEYS[task_name]


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: Any,
    task_name: str,
    max_length: int = 128,
) -> DatasetDict:
    """
    Tokenize a GLUE dataset and prepare labels for PyTorch training.
    Args:
        dataset (DatasetDict): Raw GLUE dataset splits.
        tokenizer (Any): Hugging Face tokenizer used to encode text.
        task_name (str): GLUE task identifier.
        max_length (int): Maximum encoded sequence length.
    Returns:
        DatasetDict: Tokenized dataset formatted for PyTorch.
    Algorithm:
        1. Resolve the text columns for the selected task.
        2. Tokenize single-sentence or sentence-pair batches.
        3. Rename label to labels and set tensor output columns.
    """
    sentence1_key, sentence2_key = get_task_text_keys(task_name)

    def tokenize_batch(batch: dict) -> dict:
        """
        Tokenize one mapped dataset batch.
        Args:
            batch (dict): Batch containing task text columns.
        Returns:
            dict: Tokenizer output for the batch.
        Algorithm:
            1. Read the primary sentence column.
            2. Include the secondary sentence column when present.
            3. Apply truncation, fixed padding, and max-length limits.
        """
        tokenizer_kwargs = {
            "truncation": True,
            "padding": "max_length",
            "max_length": max_length,
        }
        if sentence2_key is None:
            return tokenizer(batch[sentence1_key], **tokenizer_kwargs)
        return tokenizer(batch[sentence1_key], batch[sentence2_key], **tokenizer_kwargs)

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized_dataset


def get_eval_split(tokenized_dataset: DatasetDict, task_name: str):
    """
    Select the validation split used for evaluation.
    Args:
        tokenized_dataset (DatasetDict): Tokenized GLUE dataset splits.
        task_name (str): GLUE task identifier.
    Returns:
        Any: Validation split for the selected task.
    Algorithm:
        1. Use validation_matched for MNLI.
        2. Use validation for all other supported tasks.
        3. Return the selected split.
    """
    if task_name == "mnli":
        return tokenized_dataset["validation_matched"]
    return tokenized_dataset["validation"]

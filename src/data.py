from datasets import load_dataset


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


def load_glue_dataset(task_name: str):
    return load_dataset("glue", task_name)


def tokenize_dataset(dataset, tokenizer, task_name: str, max_length: int = 128):
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]

    def tokenize_fn(batch):
        if sentence2_key is None:
            return tokenizer(
                batch[sentence1_key],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            batch[sentence1_key],
            batch[sentence2_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def get_eval_split(tokenized_dataset, task_name: str):
    if task_name == "mnli":
        return tokenized_dataset["validation_matched"]
    return tokenized_dataset["validation"]
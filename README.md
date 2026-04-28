# Fine-Tuning Large Language Models Using LoRA

**Authors**: Maryna Ohinska, Daryna Antoniuk, Olha Kaplysh

## Videos

- [Maryna Ohinska](https://www.youtube.com/watch?v=U9YXA9MURvw&t=11s)
- [Daryna Antoniuk](https://youtu.be/YGIefP0QRbI)
- [Olha Kaplysh](https://youtu.be/cY6yw76rKeA?si=DNGYE5uHM5STgnHp)

Linear Algebra project focused on implementing Low-Rank Adaptation (LoRA) for fine-tuning RoBERTa. The project explores the mathematical foundations of parameter-efficient training through low-rank matrix decomposition, rank-performance trade-offs, and singular value analysis on GLUE tasks.

The main aim is to analyse and implement efficient fine-tuning of a large language model using the LoRA method, with an emphasis on how low-rank matrix decompositions reduce the number of trainable parameters while preserving model performance. Model quality is evaluated on tasks from the GLUE benchmark, including sentiment classification, textual entailment, sentence similarity, and linguistic acceptability.

## Project tasks

1. Study the architecture of large language models (Transformers, self-attention).
2. Analyse standard full fine-tuning as a baseline.
3. Implement LoRA fine-tuning on RoBERTa.
4. Evaluate performance against full fine-tuning on GLUE.

## Supported tasks

- SST-2
- MRPC
- CoLA
- MNLI

## Method overview

LoRA reduces the number of trainable parameters by representing the update of a weight matrix as a low-rank decomposition `ΔW = BA`. In this project, LoRA is applied to the query and value projection matrices in RoBERTa attention layers, while the original pretrained weights remain frozen.

## Project structure

```bash
project/
├── src/
│   ├── lora.py           # LoRA logic
│   ├── data.py           # GLUE loading and tokenization
│   ├── train.py          # train experiments
│   ├── evaluate.py       # evaluate an already fine-tuned checkpoint
│   └── config.py         # config loader
│
├── configs/
│   └── train.yaml
│
├── scripts/
│   ├── run_train.sh
│   └── run_report_training.py
│
├── notebooks/
│   └── rank_sweep.ipynb  # Exploratory analysis and plots
│
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Example usage

```bash
python src/train.py --task sst2 --mode full
python src/train.py --task sst2 --mode lora --rank 8
python src/evaluate.py --task sst2 --checkpoint_path outputs/sst2_full_r8/checkpoint-12630
```

## Evaluation

The project compares LoRA and full fine-tuning in terms of:

- accuracy on GLUE tasks,
- number of trainable parameters,
- GPU memory usage,
- throughput.

## Planned experiments

- Full fine-tuning baseline on RoBERTa-base.
- LoRA fine-tuning with ranks `r in {4, 8, 16, 32}`.
- Rank-sensitivity analysis.
- SVD analysis of weight update spectra.


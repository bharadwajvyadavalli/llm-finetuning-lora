# llm-finetuning-lora

An end-to-end pipeline for LoRA (Low-Rank Adaptation) fine-tuning and evaluation on large language models, specifically built around Meta’s LLaMA-2 series. This repository walks you from raw instruction datasets all the way through model training, evaluation, and deployment-ready packaging.

---

## 📖 Overview

In this project, we:

1. **Ingest & Prepare Data** — Load instruction-following datasets (e.g., Alpaca, ShareGPT), perform cleaning, formatting, and tokenization to generate model-ready examples.
2. **Baseline Verification** — Download a pre‑trained LLaMA-2 model, run sanity-check inference on a few prompts, and confirm your environment is set up correctly.
3. **LoRA Fine‑Tuning** — Apply LoRA adapters via the PEFT framework, with configurable rank, alpha, and dropout, to efficiently adapt the base model without full-weight retraining.
4. **Multi‑Task Evaluation** — Assess the fine-tuned model on multiple NLP tasks:

   * **Question Answering** (accuracy)
   * **Summarization** (ROUGE)
   * **Text Classification** (accuracy)
   * **Translation** (BLEU)
   * *Extendable to more tasks as needed.*
5. **Scalability & Reliability** —

   * Checkpointing at each epoch
   * Distributed training support (multi‑GPU via DDP)
   * Experiment tracking with Weights & Biases
   * Detailed logging and error handling
6. **DevOps Ready** —

   * Dockerfile for containerized runs
   * GitHub Actions CI for linting and import checks

---

## 🚀 Features

* **Config-Driven**: All hyperparameters and file paths live in `config/default.yaml`.
* **Modular Code**: Separate modules for data, models, evaluation, and utilities.
* **Robust Error Handling**: Try/except blocks with informative logs in each step.
* **Multi-Task Evaluation Suite**: Easily add new tasks by dropping in datasets and metrics.
* **Infrastructure**: Docker + CI pipeline for reproducibility and quality assurance.

---

## 🗂 Repository Structure

```text
llm-finetuning-lora/
├── config/
│   └── default.yaml       # All dataset paths, model names, training & eval params
├── src/
│   ├── data/              # Preprocessing scripts
│   ├── models/            # Baseline and LoRA fine-tuning logic
│   ├── evaluation/        # Multi-task evaluation suite
│   └── utils/             # Logging, trainer, DDP, checkpointing
├── scripts/               # Convenience wrappers (train.sh, evaluate.sh)
├── Dockerfile             # Container definition
├── .github/workflows/ci.yml  # CI pipeline
├── requirements.txt       # Dependencies
├── setup.cfg              # Linting rules
├── README.md
└── .gitignore
```

---

## 🛠 Prerequisites

* Python ≥3.10
* CUDA‑enabled GPU (for fine-tuning)
* Access to Hugging Face model hub (for LLaMA weights)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Edit `config/default.yaml` to:

* Point to your **raw data** (`data.input_path`) and **cache/training** files
* Choose `model.name` (e.g., `meta-llama/Llama-2-7b`)
* Tune LoRA hyperparameters under `training.lora_params`
* Define evaluation dataset paths under `evaluation.datasets`

---

## 🏃‍♂️ Usage

1. **Preprocess Data**

   ```bash
   python src/main.py --mode preprocess --config config/default.yaml
   ```

2. **Baseline Check**

   ```bash
   python src/main.py --mode baseline --config config/default.yaml
   ```

3. **Train with LoRA**

   ```bash
   python src/main.py --mode train --config config/default.yaml
   ```

4. **Evaluate**

   ```bash
   python src/main.py --mode eval --config config/default.yaml
   ```

5. **Docker**

   ```bash
   docker build -t llm-lora .
   docker run --gpus all -v $(pwd):/app llm-lora --mode train --config config/default.yaml
   ```

6. **CI** automatically runs on push/pull requests to ensure code quality.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:

* New evaluation tasks or metrics
* Support for additional datasets
* Improvements to error handling or logging

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

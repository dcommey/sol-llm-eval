# Sol-LLM-Eval: Automated Smart Contract Auditing Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive benchmark framework for evaluating **Open-Source Large Language Models (LLMs)** on smart contract vulnerability detection tasks. This repository contains the source code, datasets, and experimental results for the paper **"Automated Smart Contract Auditing: A Comparative Benchmark of Open-Source LLMs for Vulnerability Detection"** (Under Review).

## 🔬 Research Overview

We conduct a rigorous empirical evaluation of state-of-the-art open-source 7B-parameter models to determine their viability as automated auditing assistants.

**Key Findings:**
- **Qwen2.5-Coder-7B-Instruct** achieves the highest F1-score (0.803), significantly outperforming traditional static analysis (Slither) in semantic reasoning.
- **Trade-off**: While LLMs offer higher recall (96.9%) than static analysis, they exhibit higher false positive rates (18.6%), necessitating a hybrid usage strategy.
- **Efficiency**: Modern 4-bit quantized models on consumer hardware (M-series Mac) can audit contracts in ~10 seconds, enabling CI/CD integration.

**Supported Models:**
- **Qwen2.5-Coder-7B-Instruct** (Alibaba Cloud)
- **DeepSeek-Coder-7B-Instruct-v1.5** (DeepSeek AI)
- **CodeLLaMA-7B-Instruct** (Meta)
- **Mistral-7B-Instruct-v0.3** (Mistral AI)

## ⚡ Features

- **Reproducible Pipeline**: End-to-end scripts for dataset download, inference, and metric calculation.
- **Local Inference**: Uses `mlx-lm` and `transformers` for fully local execution—no API keys or external dependencies required.
- **Rigorous Evaluation**:
  - **Dataset**: SmartBugs-Curated (143 contracts, expert-annotated).
  - **Metrics**: Precision, Recall, F1-Score, TNR/FPR, McNemar's statistical tests.
  - **Analysis**: Per-vulnerability heatmaps, efficiency frontiers, and confusion matrices.
- **Professional Visualization**: Generates publication-quality PDF figures (e.g., radar charts, PR curves).

## 🛠️ Installation

### Prerequisites
- **Python 3.10+**
- **Hardware**: Apple Silicon (M1/M2/M3) or NVIDIA GPU (CUDA 11.8+). *Note: The default configuration is optimized for Apple Silicon (MLX).*

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sol-llm-eval.git
   cd sol-llm-eval
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**
   ```bash
   bash scripts/download_dataset.sh
   # Downloads and filters SmartBugs-Curated dataset to data/raw/
   ```

## 🚀 Usage

### 1. Quick Start (Test Run)
Verify your installation by running a small evaluation on 2 contracts:
```bash
python scripts/run_full_experiment.py \
    --config config/config.yaml \
    --dry-run \
    --n-samples 2 \
    --models qwen
```

### 2. Full Experiment Replication
To reproduce the paper's results (runs all 4 models on the full dataset):
```bash
python scripts/run_full_experiment.py --config config/config.yaml
```
*Estimated runtime: 3-5 hours on M3 Max.*

### 3. Generate Figures & Analysis
After inference, regenerate all plots and statistical tables:
```bash
python scripts/regenerate_figures.py
```
Output location: `paper/figures/` and `results/evaluations/`.


## 📂 Repository Structure

```
sol-llm-eval/
├── config/              # Experiment configurations (prompts, model params)
├── data/                # Dataset storage (raw & processed JSON)
├── results/             # Experimental outputs
│   ├── evaluations/     # Metrics (JSON) and statistical tests
│   └── predictions/     # Raw LLM responses
├── scripts/             # CLI entry points for experiments
├── src/                 # Core library
│   ├── evaluator.py     # Metric calculation logic
│   ├── llm_clients.py   # Model inference (MLX/HuggingFace)
│   └── visualizations.py# Matplotlib plotting code
└── tests/               # Unit tests
```

## 📄 Citation

A full BibTeX citation will be added here upon acceptance/publication of the associated paper.

If you use this codebase, please acknowledge the authors:
*Daniel Commey, Kamel Abbad, Benjamin Appiah, Lyes Khoukhi, and Garth V. Crosby.*

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **SmartBugs Project** for the curated dataset and ground truth labels.
- **MLX Community** for efficient Apple Silicon inference kernels.

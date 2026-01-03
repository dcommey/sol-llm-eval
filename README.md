# Automated Smart Contract Auditing: Open-Source LLM Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation and research artifacts for our ICBC 2025 paper: **"Automated Smart Contract Auditing: A Comparative Benchmark of Open-Source LLMs for Vulnerability Detection"**.

## Overview

We benchmark 4 state-of-the-art open-source 7B-parameter Large Language Models on their ability to detect security vulnerabilities in Solidity smart contracts:

- **Qwen2.5-Coder-7B-Instruct** - Specialized for security code review
- **DeepSeek-Coder-7B-Instruct-v1.5** - Best-in-class code generation
- **CodeLLaMA-7B-Instruct** - Meta's established code specialist
- **Mistral-7B-Instruct-v0.3** - Strong general-purpose baseline

## Key Features

- ✅ **Fully Reproducible**: All models run locally, no API keys required
- ✅ **Fair Comparison**: All models are ~7B parameters
- ✅ **Real Vulnerabilities**: Uses SmartBugs-Curated dataset with ground truth labels
- ✅ **Consumer Hardware**: Runs on 16GB RAM with 4-bit quantization
- ✅ **Comprehensive Metrics**: Precision, Recall, F1-Score, False Positive Rate

## Hardware Requirements

- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: ~30GB for models and datasets
- **GPU**: Metal-compatible GPU (M-series chips) or CUDA GPU (optional but recommended)
- **OS**: macOS (M-series), Linux, or Windows with WSL2

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sol-llm-eval.git
cd sol-llm-eval
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
bash scripts/download_dataset.sh
```

## Usage

### Quick Start (Integration Test)

Test with 2 contracts using Qwen2.5-Coder only (~10 minutes):

```bash
python scripts/run_full_experiment.py --config config/config.yaml --dry-run --n-samples 2 --models qwen
```

### Full Benchmark

Run complete benchmark on 50+ contracts with all 4 models (~3-5 hours):

```bash
python scripts/run_full_experiment.py --config config/config.yaml
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
sol-llm-eval/
├── config/
│   └── config.yaml              # Central configuration
├── paper/
│   ├── paper.tex                # IEEE conference paper
│   ├── references.bib           # Bibliography
│   ├── figures/                 # Generated figures
│   └── tables/                  # Generated LaTeX tables
├── src/
│   ├── dataset_loader.py        # SmartBugs dataset handling
│   ├── llm_clients.py           # Model inference
│   ├── prompt_engineering.py    # Standardized prompts
│   ├── evaluator.py             # Metrics calculation
│   ├── experiment_runner.py     # Orchestration
│   └── visualizations.py        # Figure generation
├── tests/                       # Unit tests
├── data/
│   ├── raw/                     # Downloaded datasets
│   └── processed/               # Filtered contracts
└── results/
    ├── predictions/             # Model outputs
    ├── evaluations/             # Metrics
    └── figures/                 # Publication figures
```

## Configuration

Edit `config/config.yaml` to customize:
- Models to evaluate
- Dataset parameters
- Inference settings (temperature, max_tokens)
- Evaluation criteria

## Results

After running experiments, results will be saved to:
- `results/evaluations/final_metrics.json` - All metrics
- `paper/tables/*.tex` - LaTeX-formatted tables
- `paper/figures/*.pdf` - Publication-quality figures

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{yourname2025llmaudit,
  title={Automated Smart Contract Auditing: A Comparative Benchmark of Open-Source LLMs for Vulnerability Detection},
  author={Your Name and Co-authors},
  booktitle={IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SmartBugs-Curated dataset maintainers
- HuggingFace for model hosting
- Open-source LLM developers (Qwen, DeepSeek, Meta, Mistral AI)

## Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com

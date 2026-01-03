# Quick Start Guide

## Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/sol-llm-eval.git
cd sol-llm-eval

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
bash scripts/download_dataset.sh
```

## Quick Test (10 minutes)

Test the pipeline with 2 contracts using only Qwen2.5-Coder:

```bash
python scripts/run_full_experiment.py \
  --config config/config.yaml \
  --dry-run \
  --n-samples 2 \
  --models qwen
```

This will:
- Load 2 sample contracts
- Run Qwen2.5-Coder inference (downloads model ~4GB on first run)
- Generate evaluation metrics
- Create sample visualizations

**Expected runtime**: ~10 minutes (including model download)

## Full Benchmark (3-5 hours)

Run complete benchmark on 50 contracts with all 4 models:

```bash
python scripts/run_full_experiment.py --config config/config.yaml
```

Results will be saved to:
- `results/evaluations/final_metrics.json` - All metrics
- `paper/tables/*.tex` - LaTeX tables for paper
- `paper/figures/*.pdf` - Publication figures

## Compile Paper

After running experiments:

```bash
cd paper
./compile.sh
```

Output: `paper/paper.pdf`

## Customization

Edit `config/config.yaml` to:
- Enable/disable specific models
- Adjust inference parameters (temperature, max_tokens)
- Change dataset sampling
- Configure evaluation metrics

## Troubleshooting

**Out of memory**: Reduce `n_samples` or disable models in config

**Model download fails**: Check internet connection, HuggingFace may be rate-limiting

**Dataset not found**: Run `bash scripts/download_dataset.sh`

For detailed documentation, see `README.md`

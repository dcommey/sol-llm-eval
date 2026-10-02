# sol-llm-eval

This repository contains the code and data for a study of local language models that screen Solidity source code.

The study compares four local models with the Slither static analyzer. A hosted model gives a reference result. The study uses three vulnerability categories:

- Reentrancy.
- Integer overflow and underflow.
- Unchecked low-level calls.

## Methods in the study

| Method | Type | Artifact |
|---|---|---|
| Qwen3.5-9B | Local model | Ollama `qwen3.5:9b`, Q4_K_M |
| Gemma 4-12B | Local model | Ollama `gemma4:12b`, Q4_K_M |
| Ministral 3-8B | Local model | Ollama `ministral-3:8b`, Q4_K_M |
| Foundation-Sec-1.1-8B | Local model | Ollama `hf.co/fdtn-ai/Foundation-Sec-1.1-8B-Instruct-Q4_K_M-GGUF:Q4_K_M`, Q4_K_M |
| Claude Sonnet 5.5 | Hosted reference | `claude-sonnet-5-5` through Claude Code 2.1.80 |
| Slither 0.11.6 | Static analyzer | Four detectors, official solc 0.4.25, 0.5.17, 0.8.27 |

The files in `results/ccnc2027/metadata/` record the exact model digests and settings.

## Data

The benchmark has 140 Solidity sources:

- 97 vulnerable files from SmartBugs-Curated, with 98 labels.
- 43 OpenZeppelin files. The study treats these files as negative for the three categories.

The scripts remove all comments from the sources before the models see them. SmartBugs-Curated puts the answers in comments.

The study also uses nine vulnerable/fixed pairs (18 sources). Each pair has one vulnerable contract and one fixed contract. An EVM test confirms the behavior of each source. These sources are in `data/fresh-ccnc2027/`.

Source revisions:

- SmartBugs-Curated: `230e649123477eff332742a59a1c7cc6dc286cab`.
- OpenZeppelin Contracts: `a83d9aabbca1ad4be17acba3e1caeca90539d3cc`.

## Main results

The values are micro-averaged F1 scores on the 140 benchmark sources. "Shared" uses only reentrancy and unchecked calls, because Slither has no general arithmetic detector.

| Method | F1, three categories | F1, shared categories | Invalid responses |
|---|---|---|---|
| Qwen3.5 | 0.688 | 0.757 | 5 of 140 |
| Gemma 4 | 0.376 | 0.407 | 1 of 140 |
| Ministral 3 | 0.482 | 0.566 | 92 of 140 |
| Foundation-Sec | 0.569 | 0.597 | 17 of 140 |
| Claude Sonnet 5.5 (hosted) | 0.843 | 0.911 | 29 of 140 |
| Slither | 0.874 | 0.952 | 3 analysis failures |

The file `results/ccnc2027/evaluations/summary.json` contains all values and 95% bootstrap intervals.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/run_ccnc_experiment.py` | Runs the local models through Ollama. Contains the prompt and the parser. |
| `scripts/run_ccnc_cloud.py` | Runs the hosted reference through Claude Code. |
| `scripts/run_ccnc_slither.py` | Runs Slither. |
| `scripts/build_fresh_suite.py` | Makes the nine vulnerable/fixed pairs and runs their EVM tests. |
| `scripts/analyze_ccnc_results.py` | Calculates all metrics and intervals. |
| `scripts/verify_foundation_template.py` | Compares the Foundation-Sec chat template with the publisher template. |
| `src/` | Shared evaluator, dataset loader, and Slither wrapper. |
| `tests/` | Unit tests. |
| `tools/` | Foundation-Sec template files and the solc-js compiler bridge. |
| `data/raw/combined_dataset.json` | Benchmark sources and labels. |
| `data/fresh-ccnc2027/` | Vulnerable/fixed pair sources. |
| `results/ccnc2027/` | Frozen inputs, raw model responses, Slither output, metadata, metrics, and validation records. |

## Requirements

The study used this environment:

- macOS on an Apple M6 computer with 24 GiB memory.
- Python 3.14.2.
- Node.js 24.
- Ollama 0.34.4.
- Go, for the template check only.
- Claude Code with a Claude subscription, for the hosted reference only.

Other environments can give different timing values. They can also give different model outputs.

## Install

1. Clone the repository:

   ```bash
   git clone https://github.com/dcommey/sol-llm-eval.git
   cd sol-llm-eval
   ```

2. Make a Python virtual environment:

   ```bash
   python3 -m venv .venv
   ```

3. Install the exact Python packages:

   ```bash
   .venv/bin/pip install -r results/ccnc2027/requirements-lock.txt
   ```

4. Install the Solidity compilers:

   ```bash
   cd tools/solc-js && npm ci && cd ../..
   ```

5. Pull the local models with Ollama. Use the tags in the table above.

6. Install the Foundation-Sec chat template. The generic model import does not include it:

   ```bash
   ollama create hf.co/fdtn-ai/Foundation-Sec-1.1-8B-Instruct-Q4_K_M-GGUF:Q4_K_M -f tools/foundationsec-native.Modelfile
   ```

7. Compare each model digest with `results/ccnc2027/metadata/main_<model>.json`. An Ollama tag can change over time.

## Check the published results

You do not need the models for this step. The commands use the frozen outputs in `results/ccnc2027/`.

1. Run the tests:

   ```bash
   .venv/bin/python -m pytest -q
   ```

2. Calculate the metrics again:

   ```bash
   .venv/bin/python scripts/analyze_ccnc_results.py
   ```

3. Compare the new `summary.json` with the version in Git:

   ```bash
   git diff --stat results/ccnc2027/evaluations/summary.json
   ```

## Run the study again

Use a new output directory. Do not write over `results/ccnc2027/`. The scripts stop if the protocol, model, or data change in an existing directory.

1. Copy the frozen datasets into the new directory:

   ```bash
   mkdir -p results/rerun/datasets
   cp results/ccnc2027/datasets/*.json results/rerun/datasets/
   ```

2. Run the local models:

   ```bash
   .venv/bin/python scripts/run_ccnc_experiment.py --root results/rerun
   .venv/bin/python scripts/run_ccnc_experiment.py --root results/rerun --suite fresh
   ```

3. Run Slither:

   ```bash
   .venv/bin/python scripts/run_ccnc_slither.py --root results/rerun
   .venv/bin/python scripts/run_ccnc_slither.py --root results/rerun --suite fresh
   ```

4. Optional: run the hosted reference. This step sends the sources to a hosted service:

   ```bash
   .venv/bin/python scripts/run_ccnc_cloud.py --root results/rerun
   .venv/bin/python scripts/run_ccnc_cloud.py --root results/rerun --suite fresh
   ```

The runners save after each source. If a run stops, start the same command again. It continues from the last saved source.

## Protocol rules

- All methods get the same system message and the same instruction.
- The local models use temperature 0, seed 42, and a limit of 2,048 output tokens. Thinking is off.
- The hosted reference also has thinking off and a 2,048-token limit. Claude Code does not let you set temperature or seed. For this reason, the study ran the hosted reference two times. The second run is in `results/ccnc2027/cloud_repeat/`.
- The parser accepts one JSON array, with or without one code fence. Other text makes the response invalid.
- The scoring counts an invalid response as an empty prediction. It also counts the invalid response separately. It never counts an invalid response as a clean result.

## Known limits

- The benchmark is public. The models possibly saw these sources during training.
- The vulnerable/fixed pairs are short synthetic contracts.
- Paths in `results/ccnc2027/` use `<repo>` for the repository root and `~` for the home directory of the original computer.
- The Git history contains an earlier version of this study. Do not use those results. They have errors in the comparison protocol.

## Licenses

The code in this repository has the MIT license. SmartBugs-Curated and OpenZeppelin Contracts have their own licenses. Read those licenses before you use the sources.

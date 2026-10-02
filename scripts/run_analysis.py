
import json
import logging
from pathlib import Path
from src.slither_baseline import run_slither_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load combined dataset
    dataset_path = Path("data/raw/combined_dataset.json")
    if not dataset_path.exists():
        logger.error("Combined dataset not found!")
        return
    
    with open(dataset_path, 'r') as f:
        contracts = json.load(f)
    
    from dataclasses import asdict
    
    # Run Slither Baseline
    logger.info("Running Slither baseline...")
    slither_results = run_slither_baseline({}, contracts)
    
    if slither_results:
        # Convert dataclasses to dicts for JSON serialization
        serializable_results = {
            'display_name': slither_results['display_name'],
            'overall_metrics': slither_results['overall_metrics'],
            'results': {k: asdict(v) for k, v in slither_results['results'].items()}
        }
        
        # Save Slither results
        with open("results/evaluations/slither_metrics_corrected.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info("Corrected Slither results saved.")
    
    # Historical LLM outputs predate benchmark-annotation scrubbing and are
    # intentionally not combined with this corrected baseline.

if __name__ == "__main__":
    main()

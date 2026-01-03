
import json
import logging
import pandas as pd
from pathlib import Path
from src.slither_baseline import run_slither_baseline
from src.statistical_analysis import calculate_all_statistics

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
        with open("results/evaluations/slither_metrics.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info("Slither results saved.")
    
    # Load existing LLM results
    metrics_path = Path("results/evaluations/final_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            llm_metrics = json.load(f)
            
        # Reconstruct predictions for statistical analysis
        # (This is a simplified reconstruction since we don't have the raw per-contract prediction objects handy in final_metrics.json
        #  BUT we can create dummy prediction objects based on confusion matrix logic if needed, 
        #  OR better: specific experiment runners usually save raw predictions. Let's look for them.)
        
        # Actually, let's look for prediction files
        pred_dir = Path("results/predictions")
        predictions_by_model = {}
        
        for model_file in pred_dir.glob("*_predictions.json"):
            model_name = model_file.stem.replace("_predictions", "")
            with open(model_file, 'r') as f:
                preds = json.load(f)
                predictions_by_model[model_name] = preds
        
        if predictions_by_model:
            logger.info("Running statistical analysis...")
            stats = calculate_all_statistics(llm_metrics, predictions_by_model, contracts)
            
            with open("results/evaluations/statistical_analysis.json", "w") as f:
                json.dump(stats, f, indent=2)
            logger.info("Statistical analysis saved.")

if __name__ == "__main__":
    main()

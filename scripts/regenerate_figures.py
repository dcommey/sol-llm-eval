
import json
import logging
from pathlib import Path
import yaml
from src.research_visualizations import PublicationVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    metrics_path = Path("results/evaluations/final_metrics.json")
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return
        
    with open(metrics_path, 'r') as f:
        results = json.load(f)
        
    visualizer = PublicationVisualizer(config)
    visualizer.generate_all(results)
    
if __name__ == "__main__":
    main()

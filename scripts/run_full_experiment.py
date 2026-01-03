#!/usr/bin/env python3
"""
Main script to run the full experiment.
"""

import argparse
import logging
import yaml
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_runner import ExperimentRunner
from src.visualizations import ResultVisualizer


def setup_logging(config: dict) -> None:
    """Configure logging."""
    log_config = config['output']['logging']
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if log_config['console']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_config['level']))
    
    # File handler
    log_file = Path(log_config['file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        handlers=[console_handler, file_handler] if log_config['console'] else [file_handler]
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run LLM Smart Contract Auditing Benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run with reduced dataset for testing'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        help='Number of samples to use (overrides config)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to run (e.g., qwen deepseek)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("LLM Smart Contract Auditing Benchmark")
    logger.info("="*80)
    
    # Override config with command line arguments
    if args.n_samples:
        config['dataset']['sampling']['n_samples'] = args.n_samples
        logger.info(f"Overriding sample size: {args.n_samples}")
    
    if args.models:
        # Disable all models first
        for model_key in config['models']:
            config['models'][model_key]['enabled'] = False
        # Enable specified models
        for model_key in args.models:
            if model_key in config['models']:
                config['models'][model_key]['enabled'] = True
                logger.info(f"Enabled model: {model_key}")
            else:
                logger.warning(f"Unknown model: {model_key}")
    
    # Run experiment
    runner = ExperimentRunner(config)
    
    try:
        results = runner.run_full_experiment()
        
        logger.info("\n" + "="*80)
        logger.info("Experiment Summary")
        logger.info("="*80)
        logger.info(f"Dataset size: {results['dataset_size']}")
        logger.info(f"Models evaluated: {', '.join(results['models_evaluated'])}")
        
        # Print results summary
        for model_key, model_results in results['evaluation_results'].items():
            logger.info(f"\n{model_results['display_name']}:")
            metrics = model_results['overall_metrics']
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall: {metrics['recall']:.3f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")
        
        # Generate visualizations
        if not args.skip_visualization:
            logger.info("\n" + "="*80)
            logger.info("Generating Visualizations")
            logger.info("="*80)
            
            visualizer = ResultVisualizer(config)
            visualizer.generate_all_visualizations(results['evaluation_results'])
        
        logger.info("\n" + "="*80)
        logger.info("Experiment Complete!")
        logger.info("="*80)
        logger.info(f"Results saved to: {config['output']['evaluations_dir']}")
        logger.info(f"Figures saved to: {config['paper']['figures_dir']}")
        logger.info(f"Tables saved to: {config['paper']['tables_dir']}")
        
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Experiment runner for executing vulnerability detection benchmark.
"""

import os
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.dataset_loader import SmartBugsDatasetLoader
from src.evaluator import VulnerabilityEvaluator

# Try to import combined dataset loader
try:
    from src.combined_dataset_loader import create_combined_dataset
    COMBINED_DATASET_AVAILABLE = True
except ImportError:
    COMBINED_DATASET_AVAILABLE = False

# Try Ollama first (preferred for Apple Silicon), fall back to HuggingFace
try:
    from src.ollama_clients import create_ollama_client, OLLAMA_MODELS
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.llm_clients import create_llm_client

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates end-to-end experiment execution."""
    
    def __init__(self, config: Dict):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_loader = SmartBugsDatasetLoader(config)
        self.evaluator = VulnerabilityEvaluator(config)
        self.output_dir = Path(config['output']['results_dir'])
        self.predictions_dir = Path(config['output']['predictions_dir'])
        self.evaluations_dir = Path(config['output']['evaluations_dir'])
        
        # Create output directories
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we should use combined dataset
        self.use_combined = config.get('dataset', {}).get('use_combined', False)
        
        # Create output directories
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_enabled_models(self) -> List[str]:
        """Get list of enabled model keys."""
        enabled = []
        for model_key, model_config in self.config['models'].items():
            if model_config.get('enabled', True):
                enabled.append(model_key)
        return enabled
    
    def _save_checkpoint(self, model_key: str, results: pd.DataFrame) -> None:
        """Save intermediate results as checkpoint."""
        if not self.config['output']['checkpoint']['enabled']:
            return
        
        checkpoint_path = self.predictions_dir / f"{model_key}_checkpoint.json"
        results.to_json(checkpoint_path, orient='records', indent=2)
        logger.info(f"Saved checkpoint for {model_key}")
    
    def _load_checkpoint(self, model_key: str) -> pd.DataFrame:
        """Load checkpoint if exists."""
        checkpoint_path = self.predictions_dir / f"{model_key}_checkpoint.json"
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint for {model_key}")
            return pd.read_json(checkpoint_path)
        return pd.DataFrame()
    
    def run_model_on_dataset(
        self,
        model_key: str,
        dataset: pd.DataFrame,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Run a single model on the entire dataset.
        
        Args:
            model_key: Model identifier (e.g., 'qwen')
            dataset: DataFrame with contracts
            resume: Whether to resume from checkpoint
            
        Returns:
            DataFrame with predictions
        """
        display_name = self.config['models'][model_key]['display_name']
        logger.info(f"Running {display_name} on {len(dataset)} contracts")
        
        # Load checkpoint if resuming
        existing_results = self._load_checkpoint(model_key) if resume else pd.DataFrame()
        
        # Safely get processed contract names (handle empty DataFrame)
        if len(existing_results) > 0 and 'contract_name' in existing_results.columns:
            processed_contracts = set(existing_results['contract_name'].tolist())
            logger.info(f"Resuming from checkpoint: {len(processed_contracts)} already processed")
        else:
            processed_contracts = set()
        
        # Create LLM client - prefer Ollama for Apple Silicon
        use_ollama = self.config.get('inference', {}).get('backend', 'auto')
        
        if use_ollama == 'ollama' or (use_ollama == 'auto' and OLLAMA_AVAILABLE and model_key in OLLAMA_MODELS):
            logger.info(f"Using Ollama backend for {model_key}")
            from src.ollama_clients import create_ollama_client
            llm_client = create_ollama_client(model_key, self.config)
        else:
            logger.info(f"Using HuggingFace backend for {model_key}")
            llm_client = create_llm_client(model_key, self.config)
        
        # Load model once
        llm_client.load_model()
        
        results = []
        checkpoint_interval = self.config['output']['checkpoint']['save_interval']
        
        try:
            for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=display_name):
                contract_name = row['contract_name']
                
                # Skip if already processed
                if contract_name in processed_contracts:
                    logger.debug(f"Skipping {contract_name} (already processed)")
                    continue
                
                contract_code = row['contract_code']
                
                # Record start time
                start_time = time.time()
                
                # Analyze contract
                try:
                    analysis_result = llm_client.analyze_contract(contract_code)
                except Exception as e:
                    logger.error(f"Failed to analyze {contract_name}: {e}")
                    analysis_result = {
                        "raw_response": "",
                        "vulnerabilities": [],
                        "num_vulnerabilities": 0,
                        "error": str(e)
                    }
                
                # Record end time
                inference_time = time.time() - start_time
                
                # Log any errors
                if analysis_result.get('error'):
                    logger.warning(f"Contract {contract_name} had error: {analysis_result['error']}")
                
                # Save result
                result = {
                    'contract_name': contract_name,
                    'model': model_key,
                    'display_name': display_name,
                    'vulnerabilities': analysis_result['vulnerabilities'],
                    'num_vulnerabilities': analysis_result['num_vulnerabilities'],
                    'raw_response': analysis_result['raw_response'],
                    'inference_time': inference_time,
                    'timestamp': time.time(),
                    'error': analysis_result.get('error')
                }
                results.append(result)
                
                # Save checkpoint periodically
                if len(results) % checkpoint_interval == 0:
                    temp_df = pd.DataFrame(results)
                    if len(existing_results) > 0:
                        temp_df = pd.concat([existing_results, temp_df], ignore_index=True)
                    self._save_checkpoint(model_key, temp_df)
        
        finally:
            # Cleanup model
            llm_client.cleanup()
        
        # Combine with existing results
        results_df = pd.DataFrame(results)
        if len(existing_results) > 0:
            results_df = pd.concat([existing_results, results_df], ignore_index=True)
        
        # Save final results
        output_path = self.predictions_dir / f"{model_key}_predictions.json"
        results_df.to_json(output_path, orient='records', indent=2)
        logger.info(f"Saved {len(results_df)} predictions for {display_name}")
        
        return results_df
    
    def run_all_models(self, dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all enabled models on dataset.
        
        Args:
            dataset: DataFrame with contracts
            
        Returns:
            Dictionary mapping model keys to prediction DataFrames
        """
        enabled_models = self._get_enabled_models()
        logger.info(f"Running {len(enabled_models)} models: {enabled_models}")
        
        all_predictions = {}
        
        for model_key in enabled_models:
            logger.info(f"\n{'='*80}\nRunning model: {model_key}\n{'='*80}")
            
            predictions = self.run_model_on_dataset(model_key, dataset)
            all_predictions[model_key] = predictions
            
            logger.info(f"Completed {model_key}")
        
        return all_predictions
    
    def evaluate_all_models(
        self,
        predictions_dict: Dict[str, pd.DataFrame],
        ground_truth: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Evaluate all models against ground truth.
        
        Args:
            predictions_dict: Dictionary of predictions per model
            ground_truth: Ground truth dataset
            
        Returns:
            Dictionary of evaluation results per model
        """
        logger.info("Evaluating all models")
        
        evaluation_results = {}
        
        for model_key, predictions in predictions_dict.items():
            display_name = self.config['models'][model_key]['display_name']
            logger.info(f"Evaluating {display_name}")
            
            # Overall metrics
            overall_metrics = self.evaluator.evaluate_predictions(predictions, ground_truth)
            
            # Per-vulnerability metrics
            vuln_types = self.config['dataset']['vulnerability_types']
            per_vuln_metrics = self.evaluator.per_vulnerability_analysis(
                predictions, ground_truth, vuln_types
            )
            
            # Confusion matrix
            conf_matrix = self.evaluator.compute_confusion_matrix(predictions, ground_truth)
            
            evaluation_results[model_key] = {
                'display_name': display_name,
                'overall_metrics': overall_metrics.to_dict(),
                'per_vulnerability_metrics': {
                    vuln_type: metrics.to_dict()
                    for vuln_type, metrics in per_vuln_metrics.items()
                },
                'confusion_matrix': conf_matrix.tolist()
            }
        
        # Save evaluation results
        output_path = self.evaluations_dir / "final_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
        
        return evaluation_results
    
    def _load_combined_dataset(self) -> pd.DataFrame:
        """Load combined dataset with vulnerable and clean contracts."""
        if not COMBINED_DATASET_AVAILABLE:
            raise RuntimeError("Combined dataset loader not available")
        
        combined_path = Path("data/raw/combined_dataset.json")
        
        if combined_path.exists():
            logger.info(f"Loading existing combined dataset from {combined_path}")
            with open(combined_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        
        # Create combined dataset
        vuln_types = self.config['dataset']['vulnerability_types']
        smartbugs_path = self.config['dataset']['dataset_path']
        oz_path = "data/raw/openzeppelin-contracts"
        
        logger.info("Creating combined dataset...")
        data = create_combined_dataset(
            smartbugs_path=smartbugs_path,
            openzeppelin_path=oz_path,
            vuln_types=vuln_types,
            n_vulnerable=98,
            n_clean=50,
            output_path=str(combined_path)
        )
        
        return pd.DataFrame(data)
    
    def run_full_experiment(self) -> Dict:
        """
        Execute complete experiment pipeline.
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting full experiment")
        
        # 1. Load dataset
        logger.info("Step 1: Loading dataset")
        
        if self.use_combined and COMBINED_DATASET_AVAILABLE:
            logger.info("Using COMBINED dataset (vulnerable + clean contracts)")
            dataset = self._load_combined_dataset()
            processed_path = Path("data/raw/combined_dataset.json")
        else:
            logger.info("Using SmartBugs dataset (vulnerable contracts only)")
            dataset = self.dataset_loader.load_dataset()
            self.dataset_loader.validate_dataset(dataset)
            processed_path = Path(self.config['dataset']['dataset_path']).parent / "processed_dataset.json"
            self.dataset_loader.save_processed_dataset(dataset, str(processed_path))
        
        # Log dataset stats
        n_vuln = len(dataset[dataset.get('is_vulnerable', True) == True]) if 'is_vulnerable' in dataset.columns else len(dataset)
        n_clean = len(dataset[dataset.get('is_vulnerable', True) == False]) if 'is_vulnerable' in dataset.columns else 0
        logger.info(f"Dataset: {len(dataset)} contracts ({n_vuln} vulnerable, {n_clean} clean)")
        
        # 2. Run all models
        logger.info("Step 2: Running all models")
        all_predictions = self.run_all_models(dataset)
        
        # 3. Evaluate predictions
        logger.info("Step 3: Evaluating predictions")
        evaluation_results = self.evaluate_all_models(all_predictions, dataset)
        
        logger.info("Full experiment complete!")
        
        return {
            'dataset_size': len(dataset),
            'n_vulnerable': n_vuln,
            'n_clean': n_clean,
            'models_evaluated': list(all_predictions.keys()),
            'evaluation_results': evaluation_results
        }

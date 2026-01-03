"""
Dataset loader for SmartBugs-Curated vulnerability dataset.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from git import Repo

logger = logging.getLogger(__name__)


class SmartBugsDatasetLoader:
    """Loader for SmartBugs-Curated dataset."""
    
    # Mapping from config vulnerability names to SmartBugs categories
    # SmartBugs-Curated uses: reentrancy, arithmetic, unchecked_low_level_calls
    VULNERABILITY_MAPPING = {
        "reentrancy": "reentrancy",
        "integer_overflow": "arithmetic",
        "unchecked_low_level_calls": "unchecked_low_level_calls"  # Direct match
    }
    
    def __init__(self, config: Dict):
        """
        Initialize dataset loader.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.dataset_path = Path(config['dataset']['dataset_path'])
        self.repo_url = config['dataset']['repo_url']
        self.vulnerability_types = config['dataset']['vulnerability_types']
        self.sampling_config = config['dataset']['sampling']
        self.filters = config['dataset']['filters']
        
    def download_dataset(self) -> None:
        """Download SmartBugs-Curated dataset if not already present."""
        if self.dataset_path.exists():
            logger.info(f"Dataset already exists at {self.dataset_path}")
            return
        
        logger.info(f"Cloning SmartBugs-Curated dataset from {self.repo_url}")
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(self.repo_url, self.dataset_path)
        logger.info("Dataset download complete")
    
    def _load_contract_metadata(self) -> pd.DataFrame:
        """
        Load contract metadata from SmartBugs structure.
        
        Returns:
            DataFrame with columns: contract_path, vulnerability_type, 
                                   contract_name, ground_truth_vulnerabilities
        """
        contracts = []
        
        # SmartBugs-Curated structure: <repo>/dataset/<vulnerability_type>/<contract_name>.sol
        # Some repos have contracts directly, others have a 'dataset' subdirectory
        possible_base_paths = [
            self.dataset_path / "dataset",  # Standard SmartBugs-Curated structure
            self.dataset_path,              # Fallback if contracts are in root
        ]
        
        base_path = None
        for path in possible_base_paths:
            if path.exists() and any(path.iterdir()):
                base_path = path
                logger.info(f"Using dataset base path: {base_path}")
                break
        
        if base_path is None:
            raise ValueError(f"Could not find valid dataset structure in {self.dataset_path}")
        
        for vuln_type in self.vulnerability_types:
            smartbugs_category = self.VULNERABILITY_MAPPING.get(vuln_type, vuln_type)
            vuln_dir = base_path / smartbugs_category
            
            if not vuln_dir.exists():
                logger.warning(f"Vulnerability directory not found: {vuln_dir}")
                continue
            
            sol_files = list(vuln_dir.glob("*.sol"))
            logger.info(f"Found {len(sol_files)} contracts in {vuln_dir}")
            
            for contract_file in sol_files:
                contracts.append({
                    'contract_path': str(contract_file),
                    'vulnerability_type': vuln_type,
                    'contract_name': contract_file.stem,
                    'ground_truth_vulnerabilities': [vuln_type]  # Known vulnerable
                })
        
        if len(contracts) == 0:
            raise ValueError(
                f"No contracts found! Check that SmartBugs-Curated is downloaded "
                f"and vulnerability types match: {self.vulnerability_types}"
            )
        
        return pd.DataFrame(contracts)
    
    def _load_contract_code(self, contract_path: str) -> str:
        """Load Solidity contract code from file."""
        with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply size and quality filters to contracts."""
        filtered_contracts = []
        
        for idx, row in df.iterrows():
            code = self._load_contract_code(row['contract_path'])
            lines = code.split('\n')
            num_lines = len([l for l in lines if l.strip()])  # Non-empty lines
            
            # Apply size filters
            if num_lines < self.filters['min_contract_size']:
                continue
            if num_lines > self.filters['max_contract_size']:
                continue
            
            # Filter test contracts if configured
            if self.filters['exclude_test_contracts']:
                if 'test' in row['contract_name'].lower():
                    continue
            
            filtered_contracts.append({
                **row.to_dict(),
                'num_lines': num_lines,
                'contract_code': code
            })
        
        return pd.DataFrame(filtered_contracts)
    
    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform stratified sampling to ensure balanced representation.
        
        Args:
            df: DataFrame with contracts
            
        Returns:
            Sampled DataFrame
        """
        n_samples = self.sampling_config['n_samples']
        min_per_type = self.sampling_config['min_per_type']
        
        sampled_dfs = []
        remaining_samples = n_samples
        
        # First, ensure minimum per type
        for vuln_type in self.vulnerability_types:
            type_contracts = df[df['vulnerability_type'] == vuln_type]
            
            if len(type_contracts) < min_per_type:
                logger.warning(
                    f"Only {len(type_contracts)} contracts found for {vuln_type}, "
                    f"less than minimum {min_per_type}"
                )
                sample_size = len(type_contracts)
            else:
                sample_size = min_per_type
            
            sampled = type_contracts.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled)
            remaining_samples -= sample_size
        
        # Distribute remaining samples proportionally
        if remaining_samples > 0:
            remaining_df = df[~df.index.isin(pd.concat(sampled_dfs).index)]
            if len(remaining_df) > 0:
                additional = remaining_df.sample(
                    n=min(remaining_samples, len(remaining_df)),
                    random_state=42
                )
                sampled_dfs.append(additional)
        
        result = pd.concat(sampled_dfs).reset_index(drop=True)
        logger.info(f"Sampled {len(result)} contracts")
        logger.info(f"Distribution: {result['vulnerability_type'].value_counts().to_dict()}")
        
        return result
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load and prepare the complete dataset.
        
        Returns:
            DataFrame with sampled contracts and their metadata
        """
        logger.info("Loading SmartBugs-Curated dataset")
        
        # Ensure dataset is downloaded
        self.download_dataset()
        
        # Load metadata
        df = self._load_contract_metadata()
        logger.info(f"Found {len(df)} contracts across {len(self.vulnerability_types)} vulnerability types")
        
        # Apply filters
        df = self._apply_filters(df)
        logger.info(f"After filtering: {len(df)} contracts")
        
        # Perform sampling
        if self.sampling_config['strategy'] == 'stratified':
            df = self._stratified_sample(df)
        else:
            # Random sampling
            n_samples = min(self.sampling_config['n_samples'], len(df))
            df = df.sample(n=n_samples, random_state=42)
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Validate that dataset is properly formatted.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_columns = [
            'contract_path', 'vulnerability_type', 'contract_name',
            'ground_truth_vulnerabilities', 'contract_code'
        ]
        
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for empty contracts
        empty_contracts = df[df['contract_code'].str.strip() == '']
        if len(empty_contracts) > 0:
            raise ValueError(f"Found {len(empty_contracts)} empty contracts")
        
        # Check vulnerability types are recognized
        unknown_types = set(df['vulnerability_type']) - set(self.vulnerability_types)
        if unknown_types:
            raise ValueError(f"Unknown vulnerability types: {unknown_types}")
        
        logger.info("Dataset validation passed")
        return True
    
    def save_processed_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed dataset to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for easy loading
        df.to_json(output_path, orient='records', indent=2)
        logger.info(f"Saved processed dataset to {output_path}")
    
    def load_processed_dataset(self, input_path: str) -> pd.DataFrame:
        """Load previously processed dataset."""
        df = pd.read_json(input_path)
        logger.info(f"Loaded processed dataset from {input_path}: {len(df)} contracts")
        return df

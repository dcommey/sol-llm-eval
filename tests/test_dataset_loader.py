"""
Tests for dataset loader.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.dataset_loader import SmartBugsDatasetLoader


def test_vulnerability_mapping():
    """Test that vulnerability type mapping works correctly."""
    loader = SmartBugsDatasetLoader({
        'dataset': {
            'dataset_path': 'data/raw/smartbugs-curated',
            'repo_url': 'https://github.com/smartbugs/smartbugs-curated.git',
            'vulnerability_types': ['reentrancy', 'integer_overflow'],
            'sampling': {'strategy': 'random', 'n_samples': 10, 'min_per_type': 5},
            'filters': {'max_contract_size': 10000, 'min_contract_size': 10, 'exclude_test_contracts': True}
        }
    })
    
    assert loader.VULNERABILITY_MAPPING['reentrancy'] == 'reentrancy'
    assert loader.VULNERABILITY_MAPPING['integer_overflow'] == 'arithmetic'


def test_validate_dataset():
    """Test dataset validation."""
    loader = SmartBugsDatasetLoader({
        'dataset': {
            'dataset_path': 'data/raw/smartbugs-curated',
            'repo_url': 'https://github.com/smartbugs/smartbugs-curated.git',
            'vulnerability_types': ['reentrancy'],
            'sampling': {'strategy': 'random', 'n_samples': 10, 'min_per_type': 5},
            'filters': {'max_contract_size': 10000, 'min_contract_size': 10, 'exclude_test_contracts': True}
        }
    })
    
    # Create valid mock dataset
    df = pd.DataFrame([{
        'contract_path': '/path/to/contract.sol',
        'vulnerability_type': 'reentrancy',
        'contract_name': 'test_contract',
        'ground_truth_vulnerabilities': ['reentrancy'],
        'contract_code': 'pragma solidity ^0.8.0; contract Test {}'
    }])
    
    assert loader.validate_dataset(df) == True


def test_validate_dataset_missing_columns():
    """Test that validation fails with missing columns."""
    loader = SmartBugsDatasetLoader({
        'dataset': {
            'dataset_path': 'data/raw/smartbugs-curated',
            'repo_url': 'https://github.com/smartbugs/smartbugs-curated.git',
            'vulnerability_types': ['reentrancy'],
            'sampling': {'strategy': 'random', 'n_samples': 10, 'min_per_type': 5},
            'filters': {'max_contract_size': 10000, 'min_contract_size': 10, 'exclude_test_contracts': True}
        }
    })
    
    # Invalid dataset missing columns
    df = pd.DataFrame([{'contract_name': 'test'}])
    
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.validate_dataset(df)

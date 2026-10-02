"""
Tests for dataset loader.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.dataset_loader import SmartBugsDatasetLoader
from src.combined_dataset_loader import (
    consolidate_duplicate_contracts,
    strip_smartbugs_annotations,
)


def test_strip_smartbugs_annotations_preserves_line_numbers():
    source = """/*
 * @vulnerable_at_lines: 12
 */
contract C {
    // <yes> <report> REENTRANCY
    function f() public {}
}
"""

    scrubbed = strip_smartbugs_annotations(source)

    assert scrubbed.count('\n') == source.count('\n')
    assert '@vulnerable_at_lines' not in scrubbed
    assert '<report>' not in scrubbed
    assert 'function f()' in scrubbed


def test_consolidate_duplicate_contracts_unions_labels():
    contracts = [
        {
            'source': 'smartbugs',
            'contract_name': 'Same',
            'contract_code': 'contract Same {\n}\n',
            'vulnerability_type': 'reentrancy',
            'ground_truth_vulnerabilities': ['reentrancy'],
        },
        {
            'source': 'smartbugs',
            'contract_name': 'Same',
            'contract_code': 'contract Same {\n\n}\n',
            'vulnerability_type': 'unchecked_low_level_calls',
            'ground_truth_vulnerabilities': ['unchecked_low_level_calls'],
        },
    ]

    consolidated = consolidate_duplicate_contracts(contracts)

    assert len(consolidated) == 1
    assert consolidated[0]['vulnerability_type'] == 'multi_label'
    assert consolidated[0]['ground_truth_vulnerabilities'] == [
        'reentrancy',
        'unchecked_low_level_calls',
    ]


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


def test_reference_negatives_use_declarations_not_names_or_comments(tmp_path):
    from src.combined_dataset_loader import load_openzeppelin_clean
    contracts = tmp_path / 'contracts'
    contracts.mkdir()
    (contracts / 'draft-IExample.sol').write_text('interface IExample { function f() external; }')
    (contracts / 'IImplementation.sol').write_text('contract IImplementation { function f() public {} }')
    (contracts / 'Ordinary.sol').write_text('/* contract Misleading {} */ interface Ordinary { function f() external; }')
    (contracts / 'ShortLibrary.sol').write_text('library ShortLibrary { function f() internal pure returns (uint) { return 1; } }')
    (contracts / 'StringDecoy.sol').write_text('import "./contract Decoy.sol"; interface StringDecoy { function f() external; }')
    (contracts / 'mocks').mkdir()
    (contracts / 'mocks' / 'Mock.sol').write_text('contract Mock {}')
    selected = load_openzeppelin_clean(str(tmp_path))
    assert {r['contract_name'] for r in selected} == {'IImplementation', 'ShortLibrary'}


def test_reference_sampling_is_order_independent_and_keeps_global_rng(tmp_path):
    import random
    from src.combined_dataset_loader import load_openzeppelin_clean
    names = ['A', 'B', 'C', 'D']
    for folder, order in [('one', names), ('two', list(reversed(names)))]:
        contracts = tmp_path / folder / 'contracts'
        contracts.mkdir(parents=True)
        for name in order:
            (contracts / f'{name}.sol').write_text(f'contract {name} {{}}')
    before = random.getstate()
    first = load_openzeppelin_clean(str(tmp_path / 'one'), 2)
    second = load_openzeppelin_clean(str(tmp_path / 'two'), 2)
    assert [r['contract_name'] for r in first] == [r['contract_name'] for r in second]
    assert len(first) == 2
    assert random.getstate() == before

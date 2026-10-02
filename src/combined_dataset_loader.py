"""
Combined dataset loader that includes both vulnerable (SmartBugs) 
and clean (OpenZeppelin) contracts for proper FPR calculation.
"""

import os
import json
import random
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

SMARTBUGS_ANNOTATION = re.compile(
    r'^\s*(?://|\*)\s*(?:@vulnerable_at_lines\s*:|<yes>\s*<report>)',
    flags=re.IGNORECASE,
)


def strip_smartbugs_annotations(code: str) -> str:
    """Remove benchmark answer annotations while preserving line numbers."""
    return '\n'.join(
        '' if SMARTBUGS_ANNOTATION.match(line) else line
        for line in code.split('\n')
    )


def consolidate_duplicate_contracts(contracts: List[Dict]) -> List[Dict]:
    """Merge semantically identical benchmark copies and union their labels."""
    consolidated = {}
    order = []
    for contract in contracts:
        normalized_code = ''.join(contract['contract_code'].split())
        key = (contract.get('source'), contract['contract_name'], normalized_code)
        if key not in consolidated:
            consolidated[key] = {**contract}
            consolidated[key]['ground_truth_vulnerabilities'] = list(
                contract.get('ground_truth_vulnerabilities', [])
            )
            order.append(key)
            continue

        existing = consolidated[key]
        existing['ground_truth_vulnerabilities'] = sorted(set(
            existing['ground_truth_vulnerabilities']
            + contract.get('ground_truth_vulnerabilities', [])
        ))
        existing['vulnerability_type'] = 'multi_label'

    return [consolidated[key] for key in order]


def load_smartbugs_vulnerable(base_path: str, vuln_types: List[str]) -> List[Dict]:
    """Load vulnerable contracts from SmartBugs-Curated."""
    contracts = []
    dataset_path = Path(base_path) / "dataset"
    
    # Map vulnerability type folders
    vuln_folder_map = {
        'reentrancy': 'reentrancy',
        'integer_overflow': 'arithmetic',  # SmartBugs uses 'arithmetic' folder
        'unchecked_low_level_calls': 'unchecked_low_level_calls'
    }
    
    for vuln_type in vuln_types:
        folder_name = vuln_folder_map.get(vuln_type, vuln_type)
        vuln_path = dataset_path / folder_name
        
        if not vuln_path.exists():
            logger.warning(f"Vulnerability folder not found: {vuln_path}")
            continue
        
        for sol_file in vuln_path.glob("*.sol"):
            try:
                with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = strip_smartbugs_annotations(f.read())
                
                contracts.append({
                    'contract_name': sol_file.stem,
                    'contract_path': str(sol_file),
                    'contract_code': code,
                    'num_lines': len(code.split('\n')),
                    'vulnerability_type': vuln_type,
                    'ground_truth_vulnerabilities': [vuln_type],
                    'is_vulnerable': True,
                    'source': 'smartbugs'
                })
            except Exception as e:
                logger.warning(f"Error loading {sol_file}: {e}")
    
    logger.info(f"Loaded {len(contracts)} vulnerable contracts from SmartBugs")
    return contracts


def load_openzeppelin_clean(base_path: str, max_contracts: int = 100) -> List[Dict]:
    """Select contract/library-containing OpenZeppelin reference-negative sources.

    Target-category negatives are an evaluation assumption, not a security audit.
    Sampling follows eligibility checks and is independent of filesystem order
    and the caller's global random state. Existing frozen datasets are unchanged.
    """
    oz_path = Path(base_path)
    if not oz_path.exists():
        logger.warning(f"OpenZeppelin path not found: {oz_path}")
        return []
    if max_contracts < 0:
        raise ValueError("max_contracts must be nonnegative")
    contracts = []
    non_code = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/')
    for sol_file in sorted(oz_path.glob("contracts/**/*.sol")):
        relative_parts = sol_file.relative_to(oz_path / "contracts").parts
        if any(part.lower() in {"mock", "mocks", "test", "tests"} for part in relative_parts[:-1]):
            continue
        if "mock" in sol_file.stem.lower() or sol_file.stem.lower().startswith("test"):
            continue
        try:
            code = sol_file.read_text(encoding="utf-8", errors="ignore")
            structural = non_code.sub(" ", code)
            if not re.search(r'\b(?:contract|library)\s+[A-Za-z_$][A-Za-z0-9_$]*', structural):
                continue
            contracts.append({
                'contract_name': sol_file.stem,
                'contract_path': str(sol_file),
                'contract_code': code,
                'num_lines': len(code.split('\n')),
                'vulnerability_type': 'none',
                'ground_truth_vulnerabilities': [],
                'is_vulnerable': False,
                'source': 'openzeppelin'
            })
        except Exception as error:
            logger.warning(f"Error loading {sol_file}: {error}")
    if len(contracts) > max_contracts:
        contracts = random.Random(42).sample(contracts, max_contracts)
    logger.info(f"Loaded {len(contracts)} OpenZeppelin contract/library reference negatives")
    return contracts


def create_combined_dataset(
    smartbugs_path: str,
    openzeppelin_path: str,
    vuln_types: List[str],
    n_vulnerable: int = 98,
    n_clean: int = 50,
    output_path: Optional[str] = None
) -> List[Dict]:
    """
    Create a balanced dataset with both vulnerable and clean contracts.
    
    Args:
        smartbugs_path: Path to SmartBugs-Curated dataset
        openzeppelin_path: Path to OpenZeppelin contracts
        vuln_types: List of vulnerability types to include
        n_vulnerable: Number of vulnerable contracts to include
        n_clean: Number of clean contracts to include
        output_path: Optional path to save the dataset
    
    Returns:
        List of contract dictionaries
    """
    # Load vulnerable contracts
    vulnerable = load_smartbugs_vulnerable(smartbugs_path, vuln_types)
    
    # Sample if needed
    if len(vulnerable) > n_vulnerable:
        random.seed(42)
        vulnerable = random.sample(vulnerable, n_vulnerable)

    vulnerable = consolidate_duplicate_contracts(vulnerable)
    
    # Load clean contracts
    clean = load_openzeppelin_clean(openzeppelin_path, n_clean)
    
    # Combine
    combined = vulnerable + clean
    random.seed(42)
    random.shuffle(combined)
    
    logger.info(f"Combined dataset: {len(vulnerable)} vulnerable + {len(clean)} clean = {len(combined)} total")
    
    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Saved combined dataset to {output_path}")
    
    return combined


def get_dataset_stats(dataset: List[Dict]) -> Dict:
    """Get statistics about the dataset."""
    stats = {
        'total': len(dataset),
        'vulnerable': sum(1 for c in dataset if c.get('is_vulnerable', False)),
        'clean': sum(1 for c in dataset if not c.get('is_vulnerable', False)),
        'by_type': {},
        'by_source': {}
    }
    
    for contract in dataset:
        vuln_type = contract.get('vulnerability_type', 'unknown')
        source = contract.get('source', 'unknown')
        
        stats['by_type'][vuln_type] = stats['by_type'].get(vuln_type, 0) + 1
        stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
    
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the combined dataset loader
    dataset = create_combined_dataset(
        smartbugs_path="data/raw/smartbugs-curated",
        openzeppelin_path="data/raw/openzeppelin-contracts",
        vuln_types=['reentrancy', 'integer_overflow', 'unchecked_low_level_calls'],
        n_vulnerable=98,
        n_clean=50,
        output_path="data/raw/combined_dataset.json"
    )
    
    stats = get_dataset_stats(dataset)
    print("\nDataset Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Vulnerable: {stats['vulnerable']}")
    print(f"  Clean: {stats['clean']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By source: {stats['by_source']}")

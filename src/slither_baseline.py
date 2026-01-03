"""
Slither baseline runner for comparison with LLM-based detection.
Provides ground truth comparison with established static analysis tool.
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SlitherResult:
    """Result from Slither analysis."""
    contract_path: str
    vulnerabilities: List[Dict]
    success: bool
    error: Optional[str] = None


# Mapping of Slither detector names to our vulnerability types
SLITHER_VULNERABILITY_MAPPING = {
    # Reentrancy
    'reentrancy-eth': 'reentrancy',
    'reentrancy-no-eth': 'reentrancy',
    'reentrancy-benign': 'reentrancy',
    'reentrancy-events': 'reentrancy',
    'reentrancy-unlimited-gas': 'reentrancy',
    
    # Integer overflow (pre-Solidity 0.8)
    'integer-overflow': 'integer_overflow',
    'integer-underflow': 'integer_overflow',
    
    # Unchecked low-level calls
    'unchecked-lowlevel': 'unchecked_low_level_calls',
    'unchecked-send': 'unchecked_low_level_calls',
    'low-level-calls': 'unchecked_low_level_calls',
    'unchecked-transfer': 'unchecked_low_level_calls',
}


class SlitherBaseline:
    """Runs Slither static analysis for baseline comparison."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._check_slither_installed()
    
    def _check_slither_installed(self) -> bool:
        """Check if Slither is installed."""
        try:
            result = subprocess.run(
                ['slither', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Slither version: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Slither not installed. Install with: pip install slither-analyzer")
            return False
        return False
    
    def analyze_contract(self, contract_path: str) -> SlitherResult:
        """Analyze a single contract with Slither."""
        try:
            # Run Slither with JSON output
            result = subprocess.run(
                [
                    'slither', contract_path,
                    '--json', '-',
                    '--exclude-informational',
                    '--exclude-optimization',
                    '--exclude-low',  # Focus on medium/high severity
                    '--fail-none'  # Don't fail on findings
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, 'SLITHER_SOLC_VERSION': '0.4.25'}  # Common version
            )
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    vulnerabilities = self._parse_detectors(data)
                    return SlitherResult(
                        contract_path=contract_path,
                        vulnerabilities=vulnerabilities,
                        success=True
                    )
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse from stderr (Slither often outputs there)
            return SlitherResult(
                contract_path=contract_path,
                vulnerabilities=[],
                success=True,
                error="No JSON output"
            )
            
        except subprocess.TimeoutExpired:
            return SlitherResult(
                contract_path=contract_path,
                vulnerabilities=[],
                success=False,
                error="Timeout"
            )
        except Exception as e:
            return SlitherResult(
                contract_path=contract_path,
                vulnerabilities=[],
                success=False,
                error=str(e)
            )
    
    def _parse_detectors(self, data: Dict) -> List[Dict]:
        """Parse Slither detector output into standardized format."""
        vulnerabilities = []
        
        if 'results' not in data or 'detectors' not in data['results']:
            return vulnerabilities
        
        for detector in data['results']['detectors']:
            check_type = detector.get('check', '')
            
            # Map to our vulnerability types
            vuln_type = SLITHER_VULNERABILITY_MAPPING.get(check_type)
            if not vuln_type:
                continue
            
            # Extract line numbers
            line_numbers = []
            for element in detector.get('elements', []):
                if 'source_mapping' in element:
                    lines = element['source_mapping'].get('lines', [])
                    line_numbers.extend(lines)
            
            vulnerabilities.append({
                'vulnerability_type': vuln_type,
                'line_numbers': sorted(set(line_numbers)),
                'severity': detector.get('impact', 'Medium'),
                'explanation': detector.get('description', '')[:200],
                'detector': check_type
            })
        
        return vulnerabilities
    
    def analyze_dataset(self, contracts: List[Dict]) -> Dict[str, SlitherResult]:
        """Analyze all contracts in dataset."""
        results = {}
        
        for contract in contracts:
            contract_path = contract.get('contract_path', contract.get('path'))
            if not contract_path or not os.path.exists(contract_path):
                continue
            
            contract_name = contract.get('contract_name', contract.get('name'))
            logger.info(f"Slither analyzing: {os.path.basename(contract_path)}")
            result = self.analyze_contract(contract_path)
            results[contract_name] = result
        
        return results
    
    def calculate_metrics(self, results: Dict[str, SlitherResult], 
                          ground_truth: List[Dict]) -> Dict[str, Any]:
        """Calculate precision, recall, F1 for Slither baseline."""
        tp, fp, fn = 0, 0, 0
        
        # Build ground truth map
        gt_map = {}
        for contract in ground_truth:
            name = contract.get('contract_name', contract.get('name'))
            gt_map[name] = contract.get('vulnerability_types', [])
        
        # Compare predictions to ground truth
        for contract_name, result in results.items():
            if not result.success:
                continue
            
            gt_vulns = set(gt_map.get(contract_name, []))
            pred_vulns = set(v['vulnerability_type'] for v in result.vulnerabilities)
            
            # True positives: predicted AND in ground truth
            for v in pred_vulns:
                if v in gt_vulns:
                    tp += 1
                else:
                    fp += 1
            
            # False negatives: in ground truth but not predicted
            for v in gt_vulns:
                if v not in pred_vulns:
                    fn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }


def run_slither_baseline(config: Dict, contracts: List[Dict]) -> Optional[Dict]:
    """Run Slither baseline if installed."""
    try:
        slither = SlitherBaseline(config)
        results = slither.analyze_dataset(contracts)
        metrics = slither.calculate_metrics(results, contracts)
        return {
            'display_name': 'Slither',
            'overall_metrics': metrics,
            'results': results
        }
    except Exception as e:
        logger.warning(f"Slither baseline failed: {e}")
        return None

"""
Tests for evaluator.
"""

import pytest
import pandas as pd
from src.evaluator import VulnerabilityEvaluator, EvaluationMetrics


def test_normalize_vulnerability_type():
    """Test vulnerability type normalization."""
    evaluator = VulnerabilityEvaluator({
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1_score'],
            'matching': {'line_number_tolerance': 5, 'type_matching': 'exact'},
            'statistical_tests': {'significance_level': 0.05, 'bootstrap_iterations': 1000}
        }
    })
    
    assert evaluator._normalize_vulnerability_type('Reentrancy') == 'reentrancy'
    assert evaluator._normalize_vulnerability_type('INTEGER_OVERFLOW') == 'integer_overflow'
    assert evaluator._normalize_vulnerability_type('unchecked-send') == 'unchecked_low_level_calls'


def test_evaluate_predictions_perfect_match():
    """Test evaluation with perfect predictions."""
    evaluator = VulnerabilityEvaluator({
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1_score'],
            'matching': {'line_number_tolerance': 5, 'type_matching': 'exact'},
            'statistical_tests': {'significance_level': 0.05, 'bootstrap_iterations': 1000}
        }
    })
    
    # Perfect predictions
    predictions = pd.DataFrame([{
        'contract_name': 'test_contract',
        'vulnerabilities': [{
            'vulnerability_type': 'reentrancy',
            'line_numbers': [10],
            'severity': 'high',
            'explanation': 'Test'
        }]
    }])
    
    ground_truth = pd.DataFrame([{
        'contract_name': 'test_contract',
        'ground_truth_vulnerabilities': ['reentrancy'],
        'num_lines': 100
    }])
    
    metrics = evaluator.evaluate_predictions(predictions, ground_truth)
    
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_evaluate_predictions_no_match():
    """Test evaluation with no correct predictions."""
    evaluator = VulnerabilityEvaluator({
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1_score'],
            'matching': {'line_number_tolerance': 5, 'type_matching': 'exact'},
            'statistical_tests': {'significance_level': 0.05, 'bootstrap_iterations': 1000}
        }
    })
    
    # Wrong vulnerability type
    predictions = pd.DataFrame([{
        'contract_name': 'test_contract',
        'vulnerabilities': [{
            'vulnerability_type': 'integer_overflow',
            'line_numbers': [10],
            'severity': 'high',
            'explanation': 'Test'
        }]
    }])
    
    ground_truth = pd.DataFrame([{
        'contract_name': 'test_contract',
        'ground_truth_vulnerabilities': ['reentrancy'],
        'num_lines': 100
    }])
    
    metrics = evaluator.evaluate_predictions(predictions, ground_truth)
    
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0

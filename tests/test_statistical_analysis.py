import pytest

from src.statistical_analysis import (
    bootstrap_f1_confidence_interval,
    paired_bootstrap_f1_difference,
)


def test_bootstrap_uses_current_ground_truth_key_and_normalizes_labels():
    predictions = [{
        'vulnerabilities': [{'vulnerability_type': 'Integer Overflow'}],
    }]
    ground_truth = [{
        'ground_truth_vulnerabilities': ['integer_overflow'],
    }]

    mean, lower, upper = bootstrap_f1_confidence_interval(
        predictions,
        ground_truth,
        n_bootstrap=20,
    )

    assert mean == pytest.approx(1.0)
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_paired_bootstrap_f1_difference_detects_better_model():
    ground_truth = [
        {'ground_truth_vulnerabilities': ['reentrancy']},
        {'ground_truth_vulnerabilities': []},
    ]
    perfect = [
        {'vulnerabilities': [{'vulnerability_type': 'Reentrancy'}]},
        {'vulnerabilities': []},
    ]
    empty = [
        {'vulnerabilities': []},
        {'vulnerabilities': []},
    ]

    result = paired_bootstrap_f1_difference(
        perfect,
        empty,
        ground_truth,
        n_bootstrap=100,
    )

    assert result['difference'] == pytest.approx(1.0)
    assert result['ci_upper'] == pytest.approx(1.0)

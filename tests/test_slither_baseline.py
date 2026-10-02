import json
from unittest.mock import patch

from src.slither_baseline import SlitherBaseline


def make_baseline():
    with patch.object(SlitherBaseline, '_check_slither_installed', return_value=True):
        return SlitherBaseline({})


def test_selects_compiler_from_pragma(tmp_path):
    baseline = make_baseline()

    legacy = tmp_path / 'Legacy.sol'
    legacy.write_text('pragma solidity ^0.4.19; contract Legacy {}')
    transitional = tmp_path / 'Transitional.sol'
    transitional.write_text('pragma solidity >=0.5.0; contract Transitional {}')
    modern = tmp_path / 'Modern.sol'
    modern.write_text('pragma solidity ^0.8.24; contract Modern {}')

    assert baseline._compiler_version_for(str(legacy)) == '0.4.25'
    assert baseline._compiler_version_for(str(transitional)) == '0.5.17'
    assert baseline._compiler_version_for(str(modern)) == '0.8.27'


def test_no_json_is_a_failure(tmp_path):
    contract = tmp_path / 'Broken.sol'
    contract.write_text('pragma solidity ^0.8.24; contract Broken {}')
    completed = type('Completed', (), {'stdout': '', 'stderr': 'compile error', 'returncode': 1})()

    with patch('src.slither_baseline.subprocess.run', return_value=completed):
        result = make_baseline().analyze_contract(str(contract))

    assert not result.success
    assert result.vulnerabilities == []
    assert 'compile error' in result.error


def test_json_failure_is_not_clean(tmp_path):
    contract = tmp_path / 'Broken.sol'
    contract.write_text('pragma solidity ^0.4.19; contract Broken {}')
    completed = type('Completed', (), {
        'stdout': json.dumps({'success': False, 'error': 'invalid compilation'}),
        'stderr': '',
        'returncode': 0,
    })()

    with patch('src.slither_baseline.subprocess.run', return_value=completed):
        result = make_baseline().analyze_contract(str(contract))

    assert not result.success
    assert result.compiler_version == '0.4.25'

import json
import pytest
from scripts.run_ccnc_experiment import parse, strip_comments
from scripts.analyze_ccnc_results import align, counts, f1

def test_comments_removed_without_changing_strings_or_line_count():
    source='string url = "https://host/*text*/"; // @vulnerable_at_lines: 1\n/* answer\ncomment */ uint a;'
    result=strip_comments(source)
    assert 'https://host/*text*/' in result
    assert '@vulnerable_at_lines' not in result and 'answer' not in result
    assert result.count('\n')==source.count('\n')

def test_parser_distinguishes_clean_from_failure_and_rejects_foreign_category():
    assert parse('[]')==([],None)
    assert parse('Not JSON')[1]
    report=dict(vulnerability_type='reentrancy',line_numbers=[1],severity='High',explanation='call before state update')
    assert parse('```json\n'+json.dumps([report])+'\n```')==([report],None)
    report['vulnerability_type']='tx_origin'
    assert parse(json.dumps([report]))[1]

def test_alignment_reorders_by_identity_and_rejects_missing_or_changed_source():
    data=[dict(contract_name='a',source_sha256='a'),dict(contract_name='b',source_sha256='b')]
    assert align(data[::-1],data)==data
    with pytest.raises(ValueError):align(data[:1],data)
    with pytest.raises(ValueError):align([data[0],dict(contract_name='b',source_sha256='c')],data)

def test_shared_category_comparison_does_not_penalize_unsupported_arithmetic():
    data=[dict(ground_truth_vulnerabilities=['reentrancy','integer_overflow'])]
    rows=[dict(vulnerabilities=[dict(vulnerability_type='reentrancy'),dict(vulnerability_type='reentrancy')])]
    assert f1(counts(rows,data).sum(axis=0))==pytest.approx(2/3)
    assert f1(counts(rows,data,['reentrancy']).sum(axis=0))==1


def test_pair_success_requires_valid_repair_response_even_when_labels_are_empty():
    from scripts.analyze_ccnc_results import pair_metrics
    data=[dict(pair_id='p',ground_truth_vulnerabilities=['reentrancy']),
          dict(pair_id='p',ground_truth_vulnerabilities=[])]
    rows=[dict(vulnerabilities=[dict(vulnerability_type='reentrancy')],parse_error=None),
          dict(vulnerabilities=[],parse_error='Malformed JSON')]
    assert pair_metrics(rows,data)['pairs_label_match']==1
    assert pair_metrics(rows,data)['pairs_exact']==0
    rows[1]['parse_error']=None
    assert pair_metrics(rows,data)['pairs_exact']==1
    rows[1]['success']=False
    assert pair_metrics(rows,data)['pairs_exact']==0

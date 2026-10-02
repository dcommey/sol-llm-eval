#!/usr/bin/env python3
"""Checkpointed, auditable local study for the CCNC revision (no historical reuse)."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = 'http://localhost:11434'
MODELS = {
    'qwen35': ('qwen3.5:9b', 'Qwen3.5-9B'),
    'gemma4': ('gemma4:12b', 'Gemma4-12B'),
    'ministral3': ('ministral-3:8b', 'Ministral3-8B'),
    'foundationsec': ('hf.co/fdtn-ai/Foundation-Sec-1.1-8B-Instruct-Q4_K_M-GGUF:Q4_K_M', 'Foundation-Sec-1.1-8B'),
}
CLASSES = ['reentrancy', 'integer_overflow', 'unchecked_low_level_calls']
SYSTEM = 'You are a smart contract security auditor with expertise in Solidity.'
INSTRUCTION = '''Analyze the Solidity source below for ONLY these vulnerability categories:
reentrancy, integer_overflow (including underflow), unchecked_low_level_calls.
Respect the Solidity pragma: checked arithmetic is the default in Solidity 0.8,
but arithmetic inside unchecked blocks can wrap. A low-level call is not itself
a vulnerability when its failure is handled. Do not report other categories.
Return ONLY a JSON array. Each issue must have vulnerability_type (one of the
three exact category names), line_numbers (array of integers), severity (string),
and explanation (string). Return [] if none of the target vulnerabilities exists.
Contract source:
```solidity
{code}
```'''
OPTIONS = dict(temperature=0, seed=42, top_p=1, top_k=40,
               repeat_penalty=1, presence_penalty=0, num_predict=2048, num_ctx=32768)
# Strings are matched before comments, so URLs and comment-like string data survive.
LEX = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/')

def strip_comments(source):
    return LEX.sub(lambda m: re.sub(r'[^\n]', ' ', m[0]) if m[0].startswith(('//', '/*')) else m[0], source)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def api(path, payload=None, timeout=600):
    response = requests.get(API+path, timeout=timeout) if payload is None else requests.post(API+path, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if result.get('error'):
        raise RuntimeError(result['error'])
    return result

def parse(raw):
    """Accept a JSON array, optionally in one Markdown code fence; reject prose/invalid schema."""
    value = raw.strip()
    if value.startswith('```') and value.endswith('```'):
        value = re.sub(r'^```(?:json)?\s*', '', value)[:-3].strip()
    try:
        reports = json.loads(value)
        if not isinstance(reports, list):
            raise ValueError('Expected JSON array')
        for report in reports:
            if not isinstance(report, dict) or report.get('vulnerability_type') not in CLASSES:
                raise ValueError('Invalid target category')
            if not isinstance(report.get('line_numbers'), list) or not all(type(n) is int for n in report['line_numbers']):
                raise ValueError('Invalid line numbers')
            if not all(isinstance(report.get(k), str) for k in ['severity', 'explanation']):
                raise ValueError('Invalid report fields')
        return reports, None
    except (ValueError, TypeError) as error:
        return [], str(error)

def prepare_main(root):
    path = root / 'datasets/main.json'
    raw = json.loads((ROOT/'data/raw/combined_dataset.json').read_text())
    data = []
    for row in raw:
        item = dict(row)
        item['contract_code'] = strip_comments(row['contract_code'])
        item['source_sha256'] = hashlib.sha256(item['contract_code'].encode()).hexdigest()
        data.append(item)
    assert len(data) == 140 and len({r['contract_name'] for r in data}) == 140
    assert sum(len(r['ground_truth_vulnerabilities']) for r in data) == 98
    assert not any('@vulnerable_at_lines' in r['contract_code'] or '<yes>' in r['contract_code'] for r in data)
    if path.exists() and json.loads(path.read_text()) != data:
        raise RuntimeError('Dataset changed after snapshot; use a new output root')
    save(path, data)
    return data

def run(root, suite, keys):
    os.chdir(ROOT)
    data = prepare_main(root) if suite == 'main' else json.loads((root/f'datasets/{suite}.json').read_text())
    protocol = dict(models=MODELS, classes=CLASSES, system=SYSTEM, instruction=INSTRUCTION,
                    options=OPTIONS, thinking=False, format_constraint=None, retries=0,
                    parsing='Strict JSON array; optional single code fence; full schema validation',
                    scoring='Deduplicated target categories; parse failures are empty predictions; transport failures stop the run')
    path = root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError('Protocol changed; use a new output root')
    save(path, protocol)
    version = api('/api/version')['version']
    for key in keys:
        tag, display = MODELS[key]
        tags = {r['name']:r for r in api('/api/tags')['models']}
        if tag not in tags:
            print(f'Waiting for local model installation: {tag}', flush=True)
            while tag not in tags:
                time.sleep(15)
                tags = {r['name']:r for r in api('/api/tags')['models']}
        info = api('/api/show', {'model':tag})
        assert 'cloud' not in tag
        metadata = dict(tag=tag, display_name=display, digest=tags[tag]['digest'],
                        details=info['details'], model_info=info.get('model_info',{}),
                        template=info.get('template'), parameters=info.get('parameters'),
                        ollama_version=version, hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                        memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                        platform=platform.platform(), protocol_sha256=digest(protocol),
                        dataset_sha256=digest(data), suite=suite)
        metadata_path=root/f'metadata/{suite}_{key}.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError(f'Run provenance changed for {key}; cannot resume')
        save(metadata_path, metadata)
        output = root/f'predictions/{suite}_{key}.json'
        rows = json.loads(output.read_text()) if output.exists() else []
        done = {r['contract_name'] for r in rows}
        assert len(done) == len(rows) and done <= {r['contract_name'] for r in data}
        if len(rows) == len(data):
            print(f'Already complete: {suite}/{key}', flush=True)
            continue
        # Load/warm the model separately so timing excludes first-load overhead.
        api('/api/chat', dict(model=tag, messages=[], keep_alive='30m', options=OPTIONS))
        print(f'RUN {suite}/{key}: {len(rows)}/{len(data)} complete', flush=True)
        try:
            for row in data:
                if row['contract_name'] in done:
                    continue
                payload = dict(model=tag, messages=[dict(role='system',content=SYSTEM),
                        dict(role='user',content=INSTRUCTION.format(code=row['contract_code']))],
                        options=OPTIONS, stream=False, keep_alive='30m')
                if 'thinking' in info.get('capabilities',[]):
                    payload['think'] = False
                start = time.perf_counter()
                response = api('/api/chat',payload)
                elapsed = time.perf_counter()-start
                if not response.get('done'):
                    raise RuntimeError('Incomplete API response')
                # Conservative input budget check: reserve 2048 for generation.
                if response.get('prompt_eval_count',0) >= OPTIONS['num_ctx']-OPTIONS['num_predict']:
                    raise RuntimeError('Input may exceed context budget; run stopped')
                raw = response['message']['content']
                reports, error = parse(raw)
                rows.append(dict(contract_name=row['contract_name'], model=key,
                        display_name=display, vulnerabilities=reports, raw_response=raw,
                        parse_error=error, inference_time=elapsed, timestamp=datetime.now(timezone.utc).isoformat(),
                        source_sha256=row['source_sha256'], done_reason=response.get('done_reason'),
                        thinking=response['message'].get('thinking',''),
                        api_metrics={k:response.get(k) for k in ['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']}))
                save(output,rows)
                if len(rows)%10 == 0 or error or len(rows)==len(data):
                    print(f'{suite}/{key} {len(rows)}/{len(data)}; last {elapsed:.1f}s; parse errors {sum(bool(r["parse_error"]) for r in rows)}',flush=True)
        finally:
            api('/api/chat',dict(model=tag,messages=[],keep_alive=0))
        print(f'COMPLETE {suite}/{key}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='results/ccnc2027')
    parser.add_argument('--suite',default='main')
    parser.add_argument('--models',nargs='+',default=list(MODELS),choices=list(MODELS))
    args=parser.parse_args()
    run((ROOT/args.root).resolve(),args.suite,args.models)

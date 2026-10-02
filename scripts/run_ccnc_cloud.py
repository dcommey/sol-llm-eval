#!/usr/bin/env python3
"""Cloud reference run: Claude Sonnet 5.5 through Claude Code headless mode (subscription).

Uses the frozen datasets, system prompt, instruction, and parser of the local study.
Claude Code replaces its own system prompt with ours (--system-prompt), exposes no tools,
loads no settings or MCP servers, and runs in an empty directory. Thinking is disabled and
output is capped at 2,048 tokens through environment variables. Temperature and seed cannot
be set in this mode; that deviation is recorded in the protocol file.

Refusals are recorded as invalid responses. Transport/API errors (rate limits, outages)
stop the run without recording a row, so a later invocation resumes from the checkpoint.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_ccnc_experiment import ROOT, SYSTEM, INSTRUCTION, parse, save

KEY = 'claude'
MODEL = 'claude-sonnet-5-5'
DISPLAY = 'Claude Sonnet 5.5'
ENV = {'MAX_THINKING_TOKENS': '0', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '2048'}
COMMAND = ['claude', '-p', '--model', MODEL, '--system-prompt', SYSTEM, '--tools', '',
           '--output-format', 'json', '--no-session-persistence', '--setting-sources', '', '--strict-mcp-config']


class TransportError(RuntimeError):
    pass


def query(prompt, cwd):
    start = time.perf_counter()
    run = subprocess.run(COMMAND, input=prompt, capture_output=True, text=True, cwd=cwd,
                         env={**os.environ, **ENV}, timeout=900)
    elapsed = time.perf_counter() - start
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError:
        raise TransportError(f'Non-JSON CLI output (exit {run.returncode}): {run.stdout[:300]} {run.stderr[:300]}')
    if result.get('is_error') and 'output token maximum' in str(result.get('result')):
        # Truncation at the 2,048-token cap: Claude Code returns no partial text.
        result = dict(result, result='', stop_reason='max_tokens', truncated_without_text=True)
        return result, elapsed
    if result.get('is_error'):
        raise TransportError(f'API error: {str(result.get("result"))[:300]}')
    models = list(result.get('modelUsage', {}))
    if models != [MODEL]:
        raise TransportError(f'Unexpected model usage: {models}')
    return result, elapsed


def run(root, suite, workers, output):
    data = json.loads((root / f'datasets/{suite}.json').read_text())
    protocol = dict(model=MODEL, display_name=DISPLAY, access='Claude Code headless mode, Claude subscription',
                    command=COMMAND, environment=ENV, system=SYSTEM, instruction=INSTRUCTION,
                    deviations='Temperature, top-p, top-k, and seed cannot be set; provider defaults apply. '
                               'Request time includes network and service latency and is not comparable to local timing.',
                    refusals='Recorded as invalid responses', retries='None for model outputs; transport errors stop the run',
                    parsing='Identical to the local study')
    ppath = root / 'cloud_protocol.json'
    if ppath.exists() and json.loads(ppath.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError('Cloud protocol changed; use a new output file')
    save(ppath, protocol)
    rows = json.loads(output.read_text()) if output.exists() else []
    done = {r['contract_name'] for r in rows}
    todo = [r for r in data if r['contract_name'] not in done]
    print(f'{suite}: {len(rows)}/{len(data)} complete, {len(todo)} to run', flush=True)
    lock = threading.Lock()
    with tempfile.TemporaryDirectory() as cwd, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(query, INSTRUCTION.format(code=r['contract_code']), cwd): r for r in todo}
        for future in as_completed(futures):
            item = futures[future]
            result, elapsed = future.result()  # TransportError propagates and stops the run
            raw = result.get('result') or ''
            stop = result.get('stop_reason')
            if stop == 'refusal':
                reports, error = [], 'refusal'
            elif result.get('truncated_without_text'):
                reports, error = [], 'truncated at output token limit (no text returned)'
            else:
                reports, error = parse(raw)
            row = dict(contract_name=item['contract_name'], model=KEY, display_name=DISPLAY,
                       vulnerabilities=reports, raw_response=raw, parse_error=error, inference_time=elapsed,
                       timestamp=datetime.now(timezone.utc).isoformat(), source_sha256=item['source_sha256'],
                       done_reason='length' if stop == 'max_tokens' else stop, thinking='',
                       api_metrics=dict(stop_reason=stop, duration_ms=result.get('duration_ms'),
                                        duration_api_ms=result.get('duration_api_ms'), usage=result.get('usage'),
                                        model_usage=result.get('modelUsage'), session_id=result.get('session_id')))
            with lock:
                rows.append(row)
                save(output, rows)
                if len(rows) % 10 == 0 or error or len(rows) == len(data):
                    print(f'{suite} {len(rows)}/{len(data)}; invalid {sum(bool(r["parse_error"]) for r in rows)}', flush=True)
    assert len(rows) == len(data)
    print(f'COMPLETE {suite} -> {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/ccnc2027')
    parser.add_argument('--suite', default='main', choices=['main', 'fresh'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--repeat', action='store_true', help='write a second run under cloud_repeat/ for agreement checks')
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    out = root / ('cloud_repeat' if args.repeat else 'predictions') / f'{args.suite}_{KEY}{"_run2" if args.repeat else ""}.json'
    run(root, args.suite, args.workers, out)

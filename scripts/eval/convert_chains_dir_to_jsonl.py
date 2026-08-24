#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert per-slide chain JSON files to eval JSONL.')
parser.add_argument('--chains-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--limit', type=int, default=0)
args = parser.parse_args()
paths = sorted(args.chains_dir.glob('*.json'), key=lambda p: p.name)
if args.limit:
    paths = paths[:args.limit]
args.output.parent.mkdir(parents=True, exist_ok=True)
count = 0
with args.output.open('w', encoding='utf-8') as f:
    for path in paths:
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            print(f'skip {path}: {exc}')
            continue
        rec = {
            'slide_id': raw.get('slide_id') or path.stem,
            'chain-of-thought': raw.get('chain-of-thought') or raw.get('qa_chain') or [],
            'node_path': raw.get('node_path') or [],
            'report': raw.get('report') or raw.get('final_report') or '',
        }
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        count += 1
print(f'wrote {count} records -> {args.output}')

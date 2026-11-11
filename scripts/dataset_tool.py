#!/usr/bin/env python3
"""Inspect an authorized WAV/CSV corpus and make reproducible, group-separated splits."""
import argparse
import collections
import csv
import hashlib
import json
import random
import sys
import wave
from pathlib import Path

REQUIRED = ('audio', 'text', 'language', 'style', 'group')

def inspect(manifest):
    base = manifest.parent.resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in REQUIRED):
            raise ValueError('CSV requires columns: ' + ', '.join(REQUIRED))
        rows = list(reader)
    if not rows:
        raise ValueError('CSV is empty')
    seen = {}
    languages, styles = collections.Counter(), collections.Counter()
    groups, seconds = set(), 0.0
    issues = []
    for i, row in enumerate(rows, 2):
        if any(not (row.get(k) or '').strip() for k in REQUIRED):
            raise ValueError(f'row {i}: required value is empty')
        path = (base / row['audio']).resolve()
        if not path.is_file():
            raise ValueError(f'row {i}: audio file missing')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'row {i}: duplicate audio content (also row {seen[digest]}); remove duplicates before splitting')
        seen[digest] = i
        try:
            with wave.open(str(path), 'rb') as w:
                rate, channels, duration = w.getframerate(), w.getnchannels(), w.getnframes() / w.getframerate()
                if w.getcomptype() != 'NONE':
                    raise ValueError('compressed WAV')
        except (wave.Error, EOFError, ValueError) as exc:
            raise ValueError(f'row {i}: invalid PCM WAV: {exc}') from exc
        if rate != 24000 or channels != 1:
            raise ValueError(f'row {i}: expected 24 kHz mono PCM WAV, got {rate} Hz, {channels} channel(s)')
        if not 1 <= duration <= 15:
            issues.append({'row': i, 'issue': 'duration outside 1–15 seconds; listen and review'})
        seconds += duration
        languages[row['language'].strip()] += 1
        styles[row['style'].strip()] += 1
        groups.add(row['group'].strip())
        row['_path'] = str(path)
        row['_group'] = row['group'].strip()
    return rows, {'clips': len(rows), 'hours': round(seconds / 3600, 3), 'groups': len(groups),
                  'languages': dict(languages), 'styles': dict(styles), 'review': issues}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path, help='CSV paths are relative to this file')
    p.add_argument('--output', type=Path, help='new output directory for report and split JSONL')
    p.add_argument('--ref-audio', type=Path, help='one fixed authorized 24 kHz mono reference WAV')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    rows, report = inspect(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.output:
        return
    if not args.ref_audio:
        raise ValueError('--ref-audio is required when producing splits')
    ref = args.ref_audio.resolve()
    if not ref.is_file():
        raise ValueError('reference audio does not exist')
    with wave.open(str(ref), 'rb') as w:
        if w.getframerate() != 24000 or w.getnchannels() != 1 or w.getcomptype() != 'NONE' or not w.getnframes():
            raise ValueError('reference must be non-empty 24 kHz mono PCM WAV')
    # Reference must be independent of every split, including by content.
    ref_hash = hashlib.sha256(ref.read_bytes()).digest()
    if any(hashlib.sha256(Path(r['_path']).read_bytes()).digest() == ref_hash for r in rows):
        raise ValueError('reference audio must be a separate recording from all dataset clips')
    groups = sorted({r['_group'] for r in rows})
    if len(groups) < 3:
        raise ValueError('at least three independent groups are required')
    random.Random(args.seed).shuffle(groups)
    n = len(groups)
    n_test = max(1, round(n * .1))
    n_val = max(1, round(n * .1))
    assignment = {g: ('test' if i < n_test else 'val' if i < n_test + n_val else 'train') for i, g in enumerate(groups)}
    args.output.mkdir(parents=True, exist_ok=False)
    counts = collections.Counter()
    split_languages = collections.defaultdict(collections.Counter)
    split_styles = collections.defaultdict(collections.Counter)
    handles = {s: (args.output / f'{s}_raw.jsonl').open('w', encoding='utf-8') for s in ('train', 'val', 'test')}
    try:
        for row in rows:
            split = assignment[row['_group']]
            item = {'audio': row['_path'], 'text': row['text'].strip(), 'ref_audio': str(ref)}
            handles[split].write(json.dumps(item, ensure_ascii=False) + '\n')
            counts[split] += 1
            split_languages[split][row['language'].strip()] += 1
            split_styles[split][row['style'].strip()] += 1
    finally:
        for f in handles.values():
            f.close()
    report.update({'seed': args.seed, 'split_clips': dict(counts), 'split_groups': dict(collections.Counter(assignment.values())),
                   'split_languages': {k: dict(v) for k, v in split_languages.items()},
                   'split_styles': {k: dict(v) for k, v in split_styles.items()}})
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Created:', args.output)

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, csv.Error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

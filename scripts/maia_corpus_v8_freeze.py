#!/usr/bin/env python3
"""Non-destructive scientific freeze for Maia Corpus v8 audit.

Run in Google Colab after mounting Google Drive. Standard library only.
Does not modify original audit configuration or SQLite database.
"""
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = 'acquisition_v8_5_001'
ROOT = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_0_3_0_Chat/corpus_v8/runs') / RUN_ID
AUDIT = ROOT / 'quality_audit_full_with_tokenizer_v1'
DB = AUDIT / 'audit_state.sqlite3'
CONFIG = AUDIT / 'audit_config.json'
TOKENIZER = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_v3/tokenizer')
FREEZE = ROOT / 'freeze_v8_001'
EXPECTED_CHUNKS = 21021
EXPECTED_DOCS = 10101231
EXPECTED_PROXY = 8180848581
EXPECTED_REAL = 7590554035
EXPECTED_INVENTORY_SHA256 = 'bd8d1a6ab4a5b59d4967ad2976bccfcd66c3d9980b9b864699a4fe7f61cd4403'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, content):
    """Never overwrite an existing freeze artifact."""
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)


def main():
    if not DB.is_file() or not CONFIG.is_file():
        raise FileNotFoundError('Audit SQLite or audit_config.json is missing.')
    previous = json.loads(CONFIG.read_text(encoding='utf-8'))
    if previous.get('run_id') != RUN_ID or previous.get('inventory_sha256') != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError('Run ID or inventory hash does not match the audited run.')
    if previous.get('real_token_count') is not True:
        raise RuntimeError('Original audit was not configured for real token counting.')
    if previous.get('tokenizer_path') != '':
        print('NOTE: Recorded tokenizer path differs from the expected historical empty string.')

    # SQLite immutable=1 avoids writing WAL or journal files, but requires a fully checkpointed database.
    # Read-only mode is safer for a DB with a live writer. Do not freeze while audit is still running.
    db_uri = f'{DB.as_uri()}?mode=ro'
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('SQLite quick_check failed.')
        stats = dict(conn.execute('''SELECT COUNT(*) chunks, COALESCE(SUM(docs),0) documents,
          COALESCE(SUM(proxy_tokens),0) proxy_tokens, COALESCE(SUM(real_tokens),0) real_tokens,
          COALESCE(SUM(duplicates),0) duplicates, COALESCE(SUM(issues),0) issues,
          SUM(CASE WHEN real_tokens IS NULL THEN 1 ELSE 0 END) missing_token_counts
          FROM completed''').fetchone())
        seen_count = conn.execute('SELECT COUNT(*) FROM seen').fetchone()[0]
        error_examples = conn.execute('SELECT COUNT(*) FROM error_examples').fetchone()[0]
        issue_kinds = conn.execute('SELECT COUNT(*) FROM issue_counts').fetchone()[0]
        rows = [dict(r) for r in conn.execute('''SELECT job_id, COUNT(*) chunks, SUM(docs) documents,
          SUM(proxy_tokens) proxy_tokens, SUM(real_tokens) real_tokens,
          SUM(duplicates) duplicates, SUM(issues) issues
          FROM completed GROUP BY job_id ORDER BY job_id''')]
    finally:
        conn.close()

    checks = {
        'chunks': stats['chunks'] == EXPECTED_CHUNKS,
        'documents': stats['documents'] == EXPECTED_DOCS,
        'proxy_tokens': stats['proxy_tokens'] == EXPECTED_PROXY,
        'real_tokens': stats['real_tokens'] == EXPECTED_REAL,
        'seen_documents': seen_count == EXPECTED_DOCS,
        'no_null_token_counts': stats['missing_token_counts'] == 0,
        'no_reported_duplicates': stats['duplicates'] == 0,
        'no_reported_issues': stats['issues'] == 0,
        'no_error_examples': error_examples == 0,
        'no_issue_kinds': issue_kinds == 0,
    }
    if not all(checks.values()):
        print(json.dumps(checks, indent=2))
        raise RuntimeError('Audit consistency check failed. No freeze created.')

    if not TOKENIZER.is_dir():
        raise FileNotFoundError(f'Tokenizer directory not found: {TOKENIZER}')
    tokenizer_files = sorted(p for p in TOKENIZER.rglob('*') if p.is_file())
    if not tokenizer_files:
        raise RuntimeError('Tokenizer directory has no files.')
    tokenizer_hashes = {str(p.relative_to(TOKENIZER)): sha256(p) for p in tokenizer_files}
    tokenizer_fingerprint = hashlib.sha256(
        json.dumps(tokenizer_hashes, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()

    language_map = {'fineweb2_pt': 'pt', 'wikipedia_pt': 'pt',
                    'fineweb_edu_en': 'en', 'wikipedia_en': 'en',
                    'fineweb2_es': 'es', 'wikipedia_es': 'es'}
    languages = defaultdict(lambda: {'documents': 0, 'real_tokens': 0, 'proxy_tokens': 0})
    for row in rows:
        language = language_map.get(row['job_id'])
        if language is None:
            raise RuntimeError(f'Unexpected dataset: {row["job_id"]}')
        for field in ('documents', 'real_tokens', 'proxy_tokens'):
            languages[language][field] += row[field]
    for language in languages:
        languages[language]['token_share'] = languages[language]['real_tokens'] / stats['real_tokens']

    # Do not claim proof that this tokenizer was used historically.
    manifest = {
        'schema_version': '1.0',
        'freeze_id': 'maia_corpus_v8_001',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': RUN_ID,
        'source_inventory_sha256_recorded': EXPECTED_INVENTORY_SHA256,
        'audit_config_original': previous,
        'audit_database_path': str(DB),
        'audit_config_sha256': sha256(CONFIG),
        'audit_database_sha256': sha256(DB),
        'audit_summary': stats,
        'seen_documents': seen_count,
        'error_examples': error_examples,
        'issue_kinds': issue_kinds,
        'checks': checks,
        'datasets': rows,
        'languages': dict(sorted(languages.items())),
        'candidate_tokenizer': {
            'path': str(TOKENIZER),
            'files_sha256': tokenizer_hashes,
            'fingerprint_sha256': tokenizer_fingerprint,
            'historical_identity_verified': False,
            'note': 'Original audit config recorded tokenizer_path as empty. Candidate tokenizer is fingerprinted now; identity during the historical audit remains unproven.'
        },
        'limitations': [
            'No independent proof of which tokenizer produced historical real-token counts.',
            'Zero duplicates means none detected by the implemented audit checks; semantic duplicates are not ruled out.',
            'No guarantee of content quality, licensing clearance, or absence of train/validation leakage.',
            'No pretraining token packing, data split, or language sampling has been performed.'
        ],
        'status': 'AUDIT_PASSED_TOKENIZER_PROVENANCE_PENDING',
    }
    report = [
        '# Maia Corpus v8 — Scientific Freeze Report', '',
        f'- Freeze: `{manifest["freeze_id"]}`',
        f'- Run: `{RUN_ID}`',
        f'- Created (UTC): `{manifest["created_utc"]}`',
        f'- Chunks: **{stats["chunks"]:,}**',
        f'- Documents: **{stats["documents"]:,}**',
        f'- Proxy tokens: **{stats["proxy_tokens"]:,}**',
        f'- Real tokens: **{stats["real_tokens"]:,}**',
        f'- Real/proxy: **{stats["real_tokens"]/stats["proxy_tokens"]:.6f}**',
        f'- Reported duplicates: **{stats["duplicates"]:,}**',
        f'- Reported issues: **{stats["issues"]:,}**',
        '- SQLite integrity: **ok**', '',
        '## Language distribution', '',
        '| Language | Documents | Real tokens | Share |',
        '|---|---:|---:|---:|',
    ]
    for lang, row in sorted(languages.items()):
        report.append(f'| {lang} | {row["documents"]:,} | {row["real_tokens"]:,} | {row["token_share"]:.2%} |')
    report += ['', '## Provenance caveat', '',
               'The original audit configuration stored `tokenizer_path: ""`. The candidate Maia Lite v3 tokenizer is fingerprinted in the manifest, but this does **not** independently prove it was the tokenizer used for the completed audit.', '',
               '## Release gate', '',
               '**Audit integrity: PASS. Pretraining release: PENDING tokenizer provenance, licensing/quality policy, deduplication scope, train/validation split, and token packing.**', '']

    # Create only after all expensive verification and hashing succeeded.
    FREEZE.mkdir(parents=True, exist_ok=True)
    manifest_path = FREEZE / 'manifest.json'
    report_path = FREEZE / 'audit_report.md'
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError(f'Freeze already exists in {FREEZE}. Do not overwrite it; inspect existing files.')
    write_once(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    write_once(report_path, '\n'.join(report))
    # A detached checksum makes accidental changes detectable, not impossible.
    checksum_path = FREEZE / 'SHA256SUMS.txt'
    write_once(checksum_path, ''.join(f'{sha256(p)}  {p.name}\n' for p in (manifest_path, report_path)))
    print('\nFREEZE CREATED:', FREEZE)
    print('Manifest:', manifest_path)
    print('Report:', report_path)
    print('Checksums:', checksum_path)
    print('STATUS:', manifest['status'])
    print('Tokenizer fingerprint:', tokenizer_fingerprint)
    print('No original audit data were modified.')

if __name__ == '__main__':
    main()

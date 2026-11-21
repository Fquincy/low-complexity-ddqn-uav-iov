"""Verify the shipped export, without loading models or research data."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    record = json.loads((ROOT / 'provenance/EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in record['files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    source = json.loads((ROOT / 'provenance/SOURCE_MAP.json').read_text(encoding='utf-8'))
    for row in source['files']:
        assert sha(ROOT / row['exported_path']) == row['exported_sha256']
        if row['byte_identical']:
            assert row['source_sha256'] == row['exported_sha256']
    common = json.loads((ROOT / 'provenance/COMMON_ENVIRONMENT_MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in common['files_sha256'].items():
        assert sha(ROOT / name) == digest
    print(json.dumps(dict(status='PUBLIC_EXPORT_HASHES_VERIFIED',
        exported_files=len(record['files_sha256']), source_mappings=len(source['files']),
        new_training=False), indent=2))


if __name__ == '__main__':
    main()

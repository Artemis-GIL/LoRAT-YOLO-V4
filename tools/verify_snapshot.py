from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    manifest = json.loads((ROOT / 'provenance/SOURCE_MANIFEST.json').read_text())
    errors = []
    for item in manifest['source_files']:
        path = ROOT / item['repository_path']
        if not path.is_file() or sha(path) != item['sha256']:
            errors.append(item['repository_path'])
    if errors:
        print(json.dumps({'status': 'FAIL', 'files': errors}, indent=2))
        return 1
    print(json.dumps({'status': 'PASS', 'verified_files': len(manifest['source_files']),
                      'frozen_test420_sources': manifest['frozen_source_count']}))
    return 0
if __name__ == '__main__':
    sys.exit(main())

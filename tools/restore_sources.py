"""Restore original source locations; dry-run by default, never overwrite differences."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    items = json.loads((ROOT / 'provenance/SOURCE_MANIFEST.json').read_text())['source_files']
    missing = []
    for item in items:
        source = ROOT / item['repository_path']
        target = Path(item['original_path'])
        if digest(source) != item['sha256']:
            raise RuntimeError(f'Archive hash mismatch: {source}')
        if target.is_symlink():
            raise RuntimeError(f'Refusing destination symlink: {target}')
        if target.exists():
            if not target.is_file() or digest(target) != item['sha256']:
                raise RuntimeError(f'Existing destination differs; not overwritten: {target}')
        else:
            missing.append((source, target))
    print(f'Missing source files: {len(missing)}; mode: {"apply" if args.apply else "dry-run"}')
    if args.apply:
        for source, target in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(source.read_bytes())
            shutil.copymode(source, target)
    print('Weights, datasets and Python environments are not restored by this tool.')
if __name__ == '__main__':
    main()

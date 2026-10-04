"""Read-only hash verification of archived sources, metadata, and packaging files."""
from pathlib import Path
import gzip,hashlib,json
R=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
m=json.loads((R/'provenance/SOURCE_MANIFEST.json').read_text())
for e in m['source_files']:
    p=R/e['repository_path'];data=p.read_bytes()
    assert sha(data)==e['stored_sha256'],str(p)
    if e.get('storage_encoding')=='gzip':data=gzip.decompress(data)
    assert sha(data)==e['sha256'],str(p)
for e in m['auxiliary_files']:
    p=R/e['repository_path'];assert sha(p.read_bytes())==e['sha256'],str(p)
files=R/'provenance/REPOSITORY_FILES.json'
if files.exists():
    for e in json.loads(files.read_text()):
        p=R/e['path'];assert sha(p.read_bytes())==e['sha256'],str(p)
print(json.dumps(dict(status='PASS',frozen_snapshots=len(m['source_files']),auxiliary_files=len(m['auxiliary_files']),repository_file_manifest=files.exists())))

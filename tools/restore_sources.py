"""Restore missing frozen sources to recorded paths; refuse conflicts. Default is preview."""
from pathlib import Path
import argparse,gzip,hashlib,json
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--prefix',default='/home/xu/')
a=p.parse_args();m=json.loads((R/'provenance/SOURCE_MANIFEST.json').read_text())
planned=[];conflicts=[];same=0
for e in m['source_files']:
    dst=Path(e['original_path'])
    if not dst.is_relative_to(Path(a.prefix)):continue
    assert dst.is_absolute() and dst.is_relative_to('/home/xu'),str(dst)
    content=(R/e['repository_path']).read_bytes()
    if e.get('storage_encoding')=='gzip':content=gzip.decompress(content)
    assert hashlib.sha256(content).hexdigest()==e['sha256']
    if dst.exists():
        if dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest()==e['sha256']:same+=1
        else:conflicts.append(str(dst))
    else:planned.append((dst,content))
print(json.dumps(dict(mode='APPLY_MISSING_ONLY' if a.apply else 'PREVIEW',unchanged=same,to_create=[str(p) for p,d in planned],conflicts=conflicts),ensure_ascii=False,indent=2))
if a.apply:
    if conflicts:raise SystemExit('Conflicts detected: nothing written. Choose a separate working tree for migration.')
    for dst,data in planned:
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(data)

"""Verify the distributed files against the release's SHA256 manifest."""
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];count=0
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
    expected,relative=line.split('  ',1);path=(root/relative).resolve();assert path.is_relative_to(root)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==expected,relative;count+=1
print('ALL_RELEASE_FILES_VERIFIED',count)

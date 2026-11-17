"""Restore fixed dependencies and load this saved candidate; never start training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

for line in (root/'FINAL_SHA256SUMS.txt').read_text().splitlines():
    expected,name=line.split('  ',1);assert sha(root/name)==expected,name
plan=json.loads((root/'plan.json').read_text())
restore=Path('<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/overnight_checkpoint_20261028T120231Z/restore_for_next_session.py')
clean=os.environ.copy();clean.pop('PYTHONPATH',None);clean.pop('PYTHONHOME',None)
subprocess.run([sys.executable,str(restore)],env=clean,check=True)
subprocess.run([sys.executable,plan['launcher'],str(root/'verify_saved_negative_candidate.py'),'--root',str(root)],env=clean,check=True)
print('Saved negative-aware candidate and exact frozen environment ready. No training, data generation or evaluation started.')
print('Read RESULTS_zh.html, summary.json, independent_delivery_audit.json and CONTINUE_HERE_zh.md in',root)

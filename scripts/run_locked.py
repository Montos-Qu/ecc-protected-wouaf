"""Launch a repository script in the original verified interpreter and package guard."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable,str(root/'environment/locked/run_wouaf_locked.py')]+sys.argv[1:]))

import json
import os
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
profile_path = root / "locked_environment_profile.json"
profile = json.loads(profile_path.read_text())
python = Path(profile["prefix"]) / "bin/python"
assert python.exists(), "Frozen environment is missing; restore it before running experiments"
environment = os.environ.copy()
environment.pop("PYTHONPATH", None)
environment.pop("PYTHONHOME", None)
environment["PYTHONNOUSERSITE"] = "1"
arguments = sys.argv[1:] or ["--check"]
raise SystemExit(subprocess.call([str(python), "-I", str(root / "wouaf_locked_guard.py"), str(profile_path)] + arguments, env=environment))

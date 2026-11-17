"""Check portable Python/package identity and execute a native product entry."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
device = sys.argv[2]
target = Path(sys.argv[3]).resolve()
if not target.is_relative_to(ROOT) or not target.is_file():
    raise SystemExit("Target is outside this checkout")
if Path(sys.prefix).resolve() != Path(receipt["prefix"]).resolve():
    raise SystemExit("Portable interpreter prefix mismatch")
if platform.python_version() != receipt["python_version"]:
    raise SystemExit("Portable Python version differs; rerun setup")
profile = ROOT / "environment/portable/profile.json"
if hashlib.sha256(profile.read_bytes()).hexdigest() != receipt["profile_sha256"]:
    raise SystemExit("Portable profile changed; rerun setup")
configuration = json.loads(profile.read_text(encoding="utf-8"))
lock = configuration["lock_files"][receipt["backend"]][receipt["system"]]
if hashlib.sha256((ROOT / lock["path"]).read_bytes()).hexdigest() != lock["sha256"]:
    raise SystemExit("Portable dependency lock changed; restore the original file and rerun setup")
installed = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
             for d in importlib.metadata.distributions()}
if installed != receipt["installed_packages"]:
    raise SystemExit("Portable package inventory changed; rerun setup")
import torch
if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this runtime. Use --device cpu or rerun setup for your NVIDIA driver.")
arguments = sys.argv[4:]
if target.name in ("infer.py", "evaluate.py"):
    arguments += ["--device", device, "--allow-unvalidated-runtime"]
elif target.name == "verify_portable.py":
    arguments += ["--device", device]
sys.argv = [str(target), *arguments]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")

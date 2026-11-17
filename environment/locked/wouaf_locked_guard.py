import importlib.metadata
import json
import os
import platform
import re
import runpy
import subprocess
import sys
from pathlib import Path


profile = json.loads(Path(sys.argv[1]).read_text())
prefix = Path(profile["prefix"])
assert Path(sys.prefix).resolve() == prefix.resolve(), "Interpreter prefix mismatch"
assert platform.python_version() == profile["python"], "Python version mismatch"
assert not os.environ.get("PYTHONPATH"), "External PYTHONPATH is forbidden"


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


installed = {normalized(distribution.metadata["Name"]): distribution.version for distribution in importlib.metadata.distributions()}
assert installed == profile["python_distributions"], "Python package inventory drift; refusing to run"
native = {record["name"]: [record["version"], record["build"]] for record in [json.loads(path.read_text()) for path in (prefix / "conda-meta").glob("*.json")]}
assert native == profile["native_builds"], "Native package build drift; refusing to run"
hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=15)
devices = [line.split(",", 1) for line in hardware.stdout.splitlines() if line.strip()]
assert devices and all(name.strip() == profile["required_gpu"] and driver.strip() == profile["required_driver"] for name, driver in devices), "Host GPU or driver drift; refusing to run"
if sys.argv[2:] == ["--check"]:
    print("LOCKED_ENVIRONMENT_CHECK_PASSED", platform.python_version(), len(installed), len(native))
    raise SystemExit(0)
assert len(sys.argv) >= 3, "A target Python script or --check is required"
target = Path(sys.argv[2]).resolve()
assert target.is_file(), str(target)
sys.argv = [str(target)] + sys.argv[3:]
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")

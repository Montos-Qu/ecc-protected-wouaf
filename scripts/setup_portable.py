"""Create an isolated Windows/Linux inference runtime without changing system Python."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "environment/portable/profile.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(command, env, capture=False):
    if capture:
        return subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True,
                              capture_output=True, text=True, encoding="utf-8")
    process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8")
    for line in process.stdout:
        if not line.startswith("WARN Skipping file for "):
            print(line, end="", flush=True)
    status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)
    return subprocess.CompletedProcess(command, status)


def nvidia_devices():
    tool = shutil.which("nvidia-smi")
    if not tool:
        for candidate in (Path("C:/Windows/System32/nvidia-smi.exe"),
                          Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"),
                          Path("/usr/lib/wsl/lib/nvidia-smi")):
            if candidate.is_file():
                tool = str(candidate)
                break
    if not tool:
        return []
    for fields in ("name,driver_version,compute_cap", "name,driver_version"):
        try:
            result = subprocess.run([tool, "--query-gpu=" + fields, "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            return []
        if result.returncode == 0:
            return [row for row in csv.reader(io.StringIO(result.stdout)) if row]
    return []


def new_cuda_architecture(row):
    if len(row) > 2:
        try:
            if float(row[2].strip()) >= 10:
                return True
        except ValueError:
            pass
    return "RTX 50" in row[0]


def bootstrap_uv(profile, system, tools):
    item = profile["uv"]["artifacts"][system]
    tools.mkdir(parents=True, exist_ok=True)
    archive = tools / item["filename"]
    if archive.exists() and sha(archive) != item["sha256"]:
        raise RuntimeError("Existing uv artifact differs; refusing overwrite")
    if not archive.exists():
        fd, temporary_name = tempfile.mkstemp(prefix="uv-", suffix=".download", dir=tools)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "ecc-wouaf-portable-setup"})
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=60) as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if sha(temporary) != item["sha256"]:
                raise RuntimeError("uv bootstrap checksum mismatch")
            temporary.rename(archive)
        finally:
            temporary.unlink(missing_ok=True)
    executable = "uv.exe" if system == "windows" else "uv"
    with zipfile.ZipFile(archive) as source:
        names = [name for name in source.namelist() if Path(name).name == executable]
        if len(names) != 1:
            raise RuntimeError("Unexpected uv wheel layout")
        binary = tools / executable
        content = source.read(names[0])
        if binary.exists():
            if binary.read_bytes() != content:
                raise RuntimeError("Existing uv executable differs")
        else:
            binary.write_bytes(content)
        for name in source.namelist():
            if "/licenses/" in name.lower() and not name.endswith("/"):
                target = tools / "licenses" / Path(name).name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(source.read(name))
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def install_backend(profile, system, backend, runtime_root, uv, env):
    prefix = runtime_root / backend
    python = prefix / ("Scripts/python.exe" if system == "windows" else "bin/python")
    if not python.exists():
        run([uv, "--quiet", "--no-config", "venv", "--python", profile["python"], "--managed-python", prefix], env)
    version = run([python, "-I", "-c", "import platform;print(platform.python_version())"], env, True).stdout.strip()
    if version != profile["python"]:
        raise RuntimeError("Existing portable environment has a different Python; choose a fresh --runtime-dir")
    lock = profile["lock_files"][backend][system]
    requirements = ROOT / lock["path"]
    if sha(requirements) != lock["sha256"]:
        raise RuntimeError("Portable dependency lock checksum mismatch")
    print("INSTALLING_PORTABLE_PACKAGES", system, backend, flush=True)
    command = [uv, "--no-progress", "--no-config", "pip", "sync", "--python", python, "--only-binary", ":all:",
               "--index-url", "https://pypi.org/simple", "--require-hashes", requirements]
    run(command, env)
    run([uv, "--quiet", "--no-config", "pip", "check", "--python", python], env)
    return prefix, python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cuda-backend", choices=("auto", "cu118", "cu128"), default="auto")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".runtime/portable")
    args = parser.parse_args()
    system = platform.system().lower()
    if system not in ("windows", "linux") or platform.machine().lower() not in ("amd64", "x86_64"):
        raise SystemExit("Portable binaries currently support 64-bit Windows and x86_64 Linux")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    runtime_root = args.runtime_dir.resolve()
    tools = runtime_root.parent / "bootstrap"
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env.update(UV_PYTHON_INSTALL_DIR=str(tools / "python"), UV_PYTHON_BIN_DIR=str(tools / "bin"),
               UV_CACHE_DIR=str(tools / "cache"), UV_NO_CONFIG="1", PYTHONNOUSERSITE="1")
    uv = bootstrap_uv(profile, system, tools)
    if not run([uv, "--version"], env, True).stdout.startswith("uv " + profile["uv"]["version"]):
        raise RuntimeError("Unexpected uv version")
    devices = nvidia_devices()
    if args.device == "cuda" and not devices:
        raise SystemExit("No working NVIDIA driver was detected. Install a supported driver or choose --device cpu")
    wants_cuda = args.device == "cuda" or args.device == "auto" and bool(devices)
    backend = "cpu"
    if wants_cuda:
        if args.cuda_backend != "auto":
            backend = args.cuda_backend
        else:
            new_architecture = any(new_cuda_architecture(row) for row in devices)
            backend = "cu128" if new_architecture else "cu118"
    candidates = [backend] + (["cpu"] if args.device == "auto" and backend != "cpu" else [])
    for selected in candidates:
        prefix, python = install_backend(profile, system, selected, runtime_root, uv, env)
        device = "cpu" if selected == "cpu" else "cuda"
        probe = "import torch,json;d=" + repr(device) + ";x=torch.ones((1,1,4,4),device=d);y=torch.nn.functional.conv2d(x,torch.ones((1,1,3,3),device=d));assert y.sum().item()==36;print(json.dumps({'device':d,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if d=='cuda' else None}))"
        try:
            observed = json.loads(run([python, "-I", "-c", probe], env, True).stdout)
            break
        except subprocess.CalledProcessError as error:
            if device != "cuda" or args.device != "auto":
                if error.stderr:
                    print(error.stderr, file=sys.stderr)
                raise
            print("CUDA_RUNTIME_UNAVAILABLE: automatic setup is selecting the CPU runtime.", flush=True)
    inventory_code = "import importlib.metadata,json,re;print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in importlib.metadata.distributions()}))"
    inventory = json.loads(run([python, "-I", "-c", inventory_code], env, True).stdout)
    for package, expected in profile["common_packages"].items():
        if inventory[package] != expected:
            raise RuntimeError("Portable dependency version mismatch: " + package)
    for package in ("torch", "torchvision"):
        if inventory[package] != profile[package] + "+" + selected:
            raise RuntimeError("Portable torch backend mismatch: " + package)
    receipt = {"runtime": "portable", "prefix": str(prefix), "python": str(python), "python_version": profile["python"],
               "profile_sha256": sha(PROFILE), "installed_packages": inventory, "backend": selected, "device": device,
               "software_verified": True, "model_validation_passed": False, "reference_runtime_verified": False,
               "system": system, "observed_device": observed, "detected_nvidia_devices": devices, "training_performed": False}
    target = ROOT / "outputs/portable_install.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("PORTABLE_SOFTWARE_READY", system, selected, python, flush=True)


if __name__ == "__main__":
    main()

"""Launch product inference in its verified, isolated portable environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("target", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = ROOT / "outputs/portable_install.json"
    if not path.exists():
        raise SystemExit("Portable environment is missing. Run: python wouaf.py setup")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    python = Path(receipt["python"])
    if not python.is_file():
        raise SystemExit("Portable interpreter was moved or removed. Rerun setup in this checkout.")
    target = args.target.resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise SystemExit("The target must be a script in this project")
    device = receipt["device"] if args.device == "auto" else args.device
    if target.name != "verify_portable.py" and not receipt.get("model_validation_passed"):
        raise SystemExit("Model self-check has not passed. Rerun: python wouaf.py setup")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    command = [str(python), "-I", "-X", "utf8", str(ROOT / "scripts/portable_guard.py"), str(path), device,
               str(target), *args.arguments]
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()

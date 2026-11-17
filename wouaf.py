"""Pretrained WOUAF product: automatic setup, default V3 inference, optional V1/V2."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
IMAGE_ASSETS = [
    "independent_source_pngs_positive_k11", "independent_source_pngs_positive_k16",
    "independent_source_pngs_generated_negative", "independent_source_pngs_natural_negative",
    "independent_encoded_jpeg_q90_444", "independent_encoded_jpeg_q90_420",
]


def host(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)


def locked(script, *arguments):
    host("scripts/run_locked.py", ROOT / script, *arguments)


def setup(args):
    if not args.download_only:
        if platform.system() != "Linux":
            raise SystemExit("The frozen executable environment targets Linux x86_64 (including Colab). Use --download-only on Windows/macOS; run setup on Linux to execute the product.")
        if sys.version_info < (3, 12):
            raise SystemExit("Run this bootstrap with host Python 3.12+. It restores the separate frozen Python 3.10.21 environment automatically.")
    asset_root = args.asset_root.resolve()
    host("scripts/download_pretrained.py")
    host("scripts/download_support.py")
    preset = "independent" if args.with_evaluation else "demo"
    host("scripts/download_public_assets.py", "--preset", preset, "--asset-root", asset_root)
    host("scripts/verify_files.py")
    if args.download_only:
        print("DOWNLOADS_VERIFIED_NOT_INSTALLED: Run setup on the documented Linux reference runtime to execute inference.")
        return
    host("scripts/materialize_assets.py", "--asset-root", asset_root, "--id", "frozen_environment")
    host("scripts/restore_environment.py")
    host("scripts/run_locked.py", "--check")
    if args.with_evaluation:
        host("scripts/materialize_assets.py", "--asset-root", asset_root, "--id", *IMAGE_ASSETS)
        host("scripts/prepare_independent_data.py")
    locked("scripts/verify_pretrained.py", "--output", ROOT / "outputs/setup_verification.json")
    receipt = ROOT / "outputs/product_install.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"ready": True, "default_version": "v3", "available_versions": ["v1", "v2", "v3"],
                                  "trained_weights_verified": True, "reference_runtime_verified": True,
                                  "independent_images_installed": args.with_evaluation,
                                  "training_performed": False}, indent=2) + "\n", encoding="utf-8")
    print("PRETRAINED_V3_PRODUCT_READY; optional versions: v1, v2. No training performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("setup", help="Download trained models and restore/check the exact archived environment")
    install.add_argument("--asset-root", type=Path, default=ROOT / "offline_assets")
    install.add_argument("--download-only", action="store_true", help="Fetch/check files without executing the Linux environment")
    install.add_argument("--with-evaluation", action="store_true", help="Also download and prepare the full independent image cohort")
    predict = commands.add_parser("predict", help="Attribute a native 512x512 image with pretrained V3 by default")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--version", choices=("v3", "v1", "v2"), default="v3")
    predict.add_argument("--code", type=int, choices=(11, 16), default=11)
    predict.add_argument("--output", type=Path)
    commands.add_parser("verify", help="Compare pretrained V1/V2/V3 with the supplied reference fixtures")
    commands.add_parser("versions", help="Describe the available trained product versions")
    evaluate = commands.add_parser("evaluate", help="Evaluate a version on the prepared independent native image cohort")
    evaluate.add_argument("--version", choices=("v3", "v1", "v2"), default="v3")
    evaluate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "setup":
        setup(args)
    elif args.command == "versions":
        print("v3 (default): PNG/JPEG444 -> trained V1; actual JPEG420 -> trained V2. One forward per image.\n"
              "v1: trained balanced extractor for all image formats.\n"
              "v2: trained JPEG420 expert for all formats; PNG/JPEG444 trade-offs.\n"
              "All versions use trained weights and fixed BCH/parity/rejection policies; none requires training.")
    elif args.command == "verify":
        locked("scripts/verify_pretrained.py", "--output", ROOT / "outputs/product_verification.json")
    elif args.command == "evaluate":
        locked("evaluate.py", "--data", ROOT / "data/independent", "--version", args.version, "--out", args.out.resolve())
    else:
        arguments = ["--image", args.image.resolve(), "--version", args.version, "--code", args.code]
        if args.output:
            arguments.extend(["--output", args.output.resolve()])
        locked("infer.py", *arguments)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)

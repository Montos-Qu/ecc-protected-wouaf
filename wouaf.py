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


def setup_reference(args):
    if args.device == "cpu" or args.cuda_backend != "auto" or args.runtime_dir:
        raise SystemExit("Reference mode uses its archived CUDA runtime. Select --runtime portable for local device/backend choices.")
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


def portable(script, device, *arguments):
    host("scripts/run_portable.py", "--device", device, ROOT / script, *arguments)


def setup(args):
    if args.runtime == "reference":
        setup_reference(args)
        return
    host("scripts/download_pretrained.py")
    host("scripts/download_support.py")
    host("scripts/verify_files.py")
    if args.download_only:
        if args.with_evaluation:
            host("scripts/download_public_assets.py", "--preset", "images", "--asset-root", args.asset_root.resolve())
        print("TRAINED_PRODUCT_FILES_VERIFIED: run setup without --download-only to create the local runtime.")
        return
    options = ["--device", args.device, "--cuda-backend", args.cuda_backend]
    if args.runtime_dir:
        options.extend(["--runtime-dir", args.runtime_dir.resolve()])
    host("scripts/setup_portable.py", *options)
    if args.with_evaluation:
        asset_root = args.asset_root.resolve()
        host("scripts/download_public_assets.py", "--preset", "images", "--asset-root", asset_root)
        host("scripts/materialize_assets.py", "--asset-root", asset_root, "--id", *IMAGE_ASSETS)
        host("scripts/prepare_independent_data.py")
    portable("scripts/verify_portable.py", "auto", "--output", ROOT / "outputs/portable_verification.json")
    runtime = json.loads((ROOT / "outputs/portable_install.json").read_text(encoding="utf-8"))
    receipt = {"ready": True, "runtime": "portable", "default_version": "v3", "available_versions": ["v1", "v2", "v3"],
               "device": runtime["device"], "backend": runtime["backend"], "trained_weights_verified": True,
               "reference_runtime_verified": False, "independent_images_installed": args.with_evaluation,
               "training_performed": False}
    (ROOT / "outputs/product_install.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("LOCAL_PRETRAINED_V3_PRODUCT_READY", runtime["device"], "optional versions: v1, v2; no training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("setup", help="Automatically prepare the local pretrained product on Windows/Linux")
    install.add_argument("--runtime", choices=("portable", "reference"), default="portable")
    install.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    install.add_argument("--cuda-backend", choices=("auto", "cu118", "cu128"), default="auto")
    install.add_argument("--runtime-dir", type=Path)
    install.add_argument("--asset-root", type=Path, default=ROOT / "offline_assets")
    install.add_argument("--download-only", action="store_true", help="Fetch/check trained product files without installing the runtime")
    install.add_argument("--with-evaluation", action="store_true", help="Also download and prepare the full independent image cohort")
    predict = commands.add_parser("predict", help="Attribute a native 512x512 image with pretrained V3 by default")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--version", choices=("v3", "v1", "v2"), default="v3")
    predict.add_argument("--code", type=int, choices=(11, 16), default=11)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--runtime", choices=("portable", "reference"), default="portable")
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    verify = commands.add_parser("verify", help="Check trained V1/V2/V3 using the installed runtime")
    verify.add_argument("--runtime", choices=("portable", "reference"), default="portable")
    verify.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    commands.add_parser("versions", help="Describe the available trained product versions")
    ui = commands.add_parser("ui", help="Open the local web interface for pretrained V3/V1/V2")
    ui.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    ui.set_defaults(runtime="portable")
    evaluate = commands.add_parser("evaluate", help="Evaluate a version on the prepared independent native image cohort")
    evaluate.add_argument("--version", choices=("v3", "v1", "v2"), default="v3")
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--runtime", choices=("portable", "reference"), default="portable")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.command not in ("setup", "versions") and args.runtime == "reference" and args.device == "cpu":
        raise SystemExit("Reference mode requires its archived CUDA profile; use --runtime portable --device cpu")
    if args.command == "setup":
        setup(args)
    elif args.command == "versions":
        print("v3 (default): PNG/JPEG444 -> trained V1; actual JPEG420 -> trained V2. One forward per image.\n"
              "v1: trained balanced extractor for all image formats.\n"
              "v2: trained JPEG420 expert for all formats; PNG/JPEG444 trade-offs.\n"
              "All versions use trained weights and fixed BCH/parity/rejection policies; none requires training.")
    elif args.command == "verify":
        if args.runtime == "reference":
            locked("scripts/verify_pretrained.py", "--output", ROOT / "outputs/product_verification.json")
        else:
            portable("scripts/verify_portable.py", args.device, "--output", ROOT / "outputs/portable_verification.json")
    elif args.command == "ui":
        arguments = ["--port", args.port]
        if args.no_browser:
            arguments.append("--no-browser")
        portable("local_ui.py", args.device, *arguments)
    elif args.command == "evaluate":
        arguments = ["--data", ROOT / "data/independent", "--version", args.version, "--out", args.out.resolve()]
        if args.runtime == "reference":
            locked("evaluate.py", *arguments)
        else:
            portable("evaluate.py", args.device, *arguments)
    else:
        arguments = ["--image", args.image.resolve(), "--version", args.version, "--code", args.code]
        if args.output:
            arguments.extend(["--output", args.output.resolve()])
        if args.runtime == "reference":
            locked("infer.py", *arguments)
        else:
            portable("infer.py", args.device, *arguments)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)

"""Restore immutable experiment predictions, demo images, ECC tables and large inventories."""
import hashlib
import json
from pathlib import Path
import zipfile
from download_pretrained import obtain, checksum

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "release_support_manifest.json").read_text(encoding="utf-8"))
    missing = []
    for item in manifest["files"]:
        target = (ROOT / item["path"]).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError("Support asset must stay inside the project")
        if target.exists():
            if target.stat().st_size != item["bytes"] or checksum(target) != item["sha256"]:
                raise RuntimeError("Existing support file differs; refusing overwrite: " + item["path"])
        else:
            missing.append(item)
    if not missing:
        print("ALL_SUPPORT_FILES_VERIFIED", len(manifest["files"]))
        return
    archive = ROOT / manifest["path"]
    obtain(manifest, archive)
    with zipfile.ZipFile(archive) as source:
        for item in missing:
            content = source.read(item["path"])
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise RuntimeError("Support member checksum mismatch: " + item["path"])
            target = ROOT / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(content)
    print("ALL_SUPPORT_FILES_RESTORED", len(missing))


if __name__ == "__main__":
    main()

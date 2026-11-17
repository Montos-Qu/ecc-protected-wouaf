"""Download the two immutable pretrained components; never initialize or train a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def obtain(item, destination):
    """Verify an existing file or atomically install a verified download."""
    if destination.exists():
        if destination.stat().st_size != item["bytes"] or checksum(destination) != item["sha256"]:
            raise RuntimeError("Existing checkpoint differs; refusing overwrite: " + str(destination))
        print("PRETRAINED_VERIFIED", destination.name)
        return
    if not item["url"].startswith("https://github.com/"):
        raise ValueError("Expected an HTTPS GitHub Release asset URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "ecc-protected-wouaf-v3"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        if temporary.stat().st_size != item["bytes"] or checksum(temporary) != item["sha256"]:
            raise RuntimeError("Downloaded checkpoint checksum mismatch: " + destination.name)
        if destination.exists():
            raise RuntimeError("Destination appeared during download; refusing overwrite")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("PRETRAINED_DOWNLOADED_AND_VERIFIED", destination.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/pretrained_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        target = (ROOT / item["path"]).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError("Checkpoint destination must stay inside the project")
        obtain(item, target)


if __name__ == "__main__":
    main()

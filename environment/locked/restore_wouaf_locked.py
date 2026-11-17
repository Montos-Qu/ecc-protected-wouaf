import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


root = Path(__file__).resolve().parent
profile = json.loads((root / "locked_environment_profile.json").read_text())
prefix = Path(profile["prefix"])
assert prefix == Path("/content/strict_wouaf_restore/env"), "This archive requires its original fixed prefix"
inventory = json.loads((root / "frozen_environment_file_inventory.json").read_text())


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


assert digest(root / "frozen_environment_file_inventory.json") == profile["file_inventory_sha256"], "Inventory checksum mismatch"


def verify_files():
    seen = set()
    for path in prefix.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.is_dir() and not path.is_symlink():
            continue
        relative = str(path.relative_to(prefix))
        seen.add(relative)
        expected = inventory[relative]
        if path.is_symlink():
            assert expected == {"symlink": os.readlink(path)}, relative
        else:
            assert digest(path) == expected["sha256"], relative
    assert seen == set(inventory), "Environment contains missing or extra files"


def exact_safe_filter(member, destination):
    filtered = tarfile.data_filter(member, destination)
    if member.issym():
        relative = str(Path(member.name).relative_to("env"))
        assert inventory[relative] == {"symlink": member.linkname}, relative
        link_path = prefix / relative
        link_target = (link_path.parent / member.linkname).resolve()
        assert link_target.is_relative_to(prefix.resolve()), relative
        filtered = filtered.replace(linkname=member.linkname)
    return filtered


if prefix.exists() and any(prefix.iterdir()):
    print("VERIFYING_EXISTING_ENVIRONMENT_NO_OVERWRITE", flush=True)
    verify_files()
else:
    archive_path = root / profile["archive_name"]
    assert digest(archive_path) == profile["archive_sha256"], "Archive checksum mismatch"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (prefix.parent / member.name).resolve()
            assert target.is_relative_to(prefix.resolve()), member.name
        archive.extractall(prefix.parent, members=members, filter=exact_safe_filter)
    verify_files()
code_zip = root / "baseline_inference_sources.zip"
assert digest(code_zip) == profile["source_archive_sha256"]
code_root = prefix.parent / "code"
with zipfile.ZipFile(code_zip) as archive:
    for name in archive.namelist():
        target = (code_root / name).resolve()
        assert target.is_relative_to(code_root.resolve())
        content = archive.read(name)
        if target.exists():
            assert target.read_bytes() == content, str(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
result = subprocess.run([sys.executable, str(root / "run_wouaf_locked.py"), "--check"], check=True)
print("FROZEN_ENVIRONMENT_RESTORED_AND_ALL_FILES_VERIFIED", len(inventory), flush=True)

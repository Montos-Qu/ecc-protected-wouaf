"""Download public V3 environment/evaluation assets without a Drive login or extra packages."""
import argparse
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and (attrs.get("id") == "download-form" or "/download" in attrs.get("action", "")):
            self.action = attrs.get("action")
            self.active = True
        if tag == "input" and self.active and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def open_public_file(file_id, probe=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError("Invalid Drive file id")
    # A fresh anonymous cookie jar is used. Browser credentials are never read.
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": file_id, "export": "download", "confirm": "t"})
    headers = {"User-Agent": "ecc-wouaf-public-reproduction"}
    if probe:
        headers["Range"] = "bytes=0-65535"
    for _ in range(3):
        response = opener.open(urllib.request.Request(url, headers=headers), timeout=60)
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            return response
        with response:
            page = response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
        parser = DownloadForm()
        parser.feed(page)
        if not parser.action or parser.fields.get("id") != file_id:
            raise RuntimeError("Drive did not return a public download. Check access or download quota; do not save the HTML as data.")
        target = urllib.parse.urljoin(url, parser.action)
        parsed = urllib.parse.urlparse(target)
        if parsed.scheme != "https" or parsed.hostname not in ("drive.google.com", "drive.usercontent.google.com"):
            raise RuntimeError("Unexpected Drive confirmation destination")
        url = target.split("?", 1)[0] + "?" + urllib.parse.urlencode(parser.fields)
    raise RuntimeError("Drive confirmation did not produce a file")


def obtain(part, root, probe=False):
    target = (root / part["path"]).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Download path must stay inside the asset directory")
    if probe:
        with open_public_file(part["file_id"], probe=True) as source:
            block = source.read(65536)
            if not block or block.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise RuntimeError("Expected file bytes, received HTML or empty content")
            print("ANONYMOUS_FILE_BYTES_ACCESSIBLE", part["path"], len(block), flush=True)
        return
    if target.exists():
        if target.stat().st_size != part["bytes"] or checksum(target) != part["sha256"]:
            raise RuntimeError("Existing part differs; refusing overwrite: " + str(target))
        print("PUBLIC_PART_VERIFIED", part["path"], flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".download", dir=target.parent)
    temporary = Path(name)
    actual = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(fd, "wb") as output, open_public_file(part["file_id"]) as source:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > part["bytes"]:
                    raise RuntimeError("Download exceeds the recorded size")
                actual.update(block)
                output.write(block)
        if count != part["bytes"] or actual.hexdigest() != part["sha256"]:
            raise RuntimeError("Public asset SHA256 or size mismatch: " + part["path"])
        if target.exists():
            raise RuntimeError("Destination appeared during download; refusing overwrite")
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    print("ANONYMOUS_DOWNLOAD_SHA256_VERIFIED", part["path"], count, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("demo", "images", "independent"), default="demo")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "offline_assets")
    parser.add_argument("--check-links", action="store_true", help="Read a small anonymous sample without saving entire files")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "public_assets.json").read_text(encoding="utf-8"))
    parts = [part for name in manifest["presets"][args.preset] for part in manifest["assets"][name]["parts"]]
    print("PUBLIC_ASSET_PLAN", args.preset, len(parts), sum(part["bytes"] for part in parts), flush=True)
    for part in parts:
        obtain(part, args.asset_root, probe=args.check_links)
    print("PUBLIC_ASSET_LINKS_PASSED" if args.check_links else "ALL_PUBLIC_ASSET_PARTS_VERIFIED", len(parts), flush=True)


if __name__ == "__main__":
    main()

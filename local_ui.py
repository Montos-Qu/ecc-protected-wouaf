"""Serve the pretrained product through a loopback-only local web interface."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web"
MAX_UPLOAD = 10 * 1024 * 1024
SAMPLES = {
    "positive_png": "demo/images/png8_control/positive_k11/10000000.png",
    "positive_jpeg": "demo/images/jpeg_q90_420/positive_k11/10000000.jpg",
    "negative_png": "demo/images/png8_control/generated_negative/10000000.png",
}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, device):
        super().__init__(address, Handler)
        self.device = device
        self.session_token = secrets.token_hex(32)
        self.inference_lock = threading.Lock()
        self.predictors = {}
        self.extractors = {}
        self.url = "http://127.0.0.1:" + str(self.server_port)
        self.checkout_id = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:16]

    def predict(self, data, version, code, filename, resize="native"):
        from ecc_wouaf import Predictor
        from image_preprocessing import prepare_image, predict_prepared
        prepared = prepare_image(data, resize)
        with self.inference_lock:
            if version not in self.predictors:
                predictor = Predictor(version, self.device, strict=False)
                # All versions use the same two immutable extractor components.
                predictor.models = self.extractors
                self.predictors[version] = predictor
            result = predict_prepared(self.predictors[version], prepared, code, filename)
        preview = None
        if prepared.preprocessing and prepared.preprocessing["applied"]:
            preview = "data:image/png;base64," + base64.b64encode(prepared.encoded).decode("ascii")
        return result, preview


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, message, *arguments):
        print(message % arguments, flush=True)

    def reply(self, status, data, content_type="application/json; charset=utf-8", attachment=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="' + attachment + '"')
        self.end_headers()
        self.wfile.write(data)

    def local_request(self, mutation=False):
        allowed = {"127.0.0.1:" + str(self.server.server_port), "localhost:" + str(self.server.server_port)}
        if self.headers.get("Host") not in allowed:
            self.reply(403, {"error": "local_access_only"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in allowed}:
            self.reply(403, {"error": "local_access_only"})
            return False
        token = self.headers.get("X-WOUAF-Token", "")
        if mutation and (not token.isascii() or not secrets.compare_digest(token, self.server.session_token)):
            self.reply(403, {"error": "session_expired"})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/health":
            self.reply(200, {"application": "ecc-wouaf-local-ui", "version": "3.0.4", "device": self.server.device,
                             "default_model": "v3", "url": self.server.url, "ready": True,
                             "checkout_id": self.server.checkout_id})
        elif path in ("/", "/app.js", "/style.css"):
            filename, content_type = {"/": ("index.html", "text/html; charset=utf-8"),
                                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                                      "/style.css": ("style.css", "text/css; charset=utf-8")}[path]
            data = (STATIC / filename).read_bytes()
            if path == "/":
                data = data.replace(b"__SESSION_TOKEN__", self.server.session_token.encode("ascii"))
            self.reply(200, data, content_type)
        elif path == "/api/sample":
            sample = parse_qs(parsed.query).get("id", [""])[0]
            if sample not in SAMPLES:
                self.reply(404, {"error": "not_found"})
                return
            target = ROOT / SAMPLES[sample]
            if not target.is_file():
                self.reply(404, {"error": "sample_missing"})
                return
            self.reply(200, target.read_bytes(), "image/jpeg" if target.suffix == ".jpg" else "image/png")
        elif re.fullmatch(r"/api/results/prediction_[0-9a-f]{32}\.json", path):
            target = ROOT / "outputs/ui" / path.rsplit("/", 1)[1]
            if target.is_file():
                self.reply(200, target.read_bytes(), attachment=target.name)
            else:
                self.reply(404, {"error": "not_found"})
        else:
            self.reply(404, {"error": "not_found"})

    def do_POST(self):
        if not self.local_request(mutation=True):
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/shutdown":
            self.reply(200, {"stopped": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if parsed.path != "/api/predict":
            self.reply(404, {"error": "not_found"})
            return
        query = parse_qs(parsed.query)
        version = query.get("version", ["v3"])[0]
        code = query.get("code", ["11"])[0]
        resize = query.get("resize", ["native"])[0]
        valid_configuration = version in ("v1", "v2", "v3") and code in ("11", "16") and resize in ("native", "fit")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_UPLOAD:
            self.reply(413, {"error": "image_too_large"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/octet-stream":
            self.reply(415, {"error": "invalid_upload"})
            return
        self.connection.settimeout(30)
        try:
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError("invalid_upload")
            if not valid_configuration:
                raise ValueError("invalid_configuration")
            filename = unquote(self.headers.get("X-Filename", "uploaded_image"))
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:240]
            started = time.perf_counter()
            result, preview = self.server.predict(data, version, int(code), filename, resize)
            elapsed = round((time.perf_counter() - started) * 1000)
            identifier = "prediction_" + secrets.token_hex(16) + ".json"
            target = ROOT / "outputs/ui" / identifier
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            self.reply(200, {"result": result, "elapsed_ms": elapsed, "download_url": "/api/results/" + identifier,
                             "processed_preview": preview})
        except ValueError as error:
            known = {"unsupported_format", "wrong_dimensions", "invalid_image", "invalid_upload", "too_many_pixels", "invalid_configuration"}
            self.reply(400, {"error": str(error) if str(error) in known else "invalid_image"})
        except TimeoutError:
            self.reply(408, {"error": "upload_timeout"})
        except Exception as error:
            print("INFERENCE_ERROR", type(error).__name__, str(error), flush=True)
            self.reply(500, {"error": "inference_failed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    try:
        server = LocalServer(("127.0.0.1", args.port), args.device)
    except OSError:
        try:
            url = "http://127.0.0.1:" + str(args.port)
            with urllib.request.urlopen(url + "/api/health", timeout=2) as source:
                health = json.load(source)
            checkout_id = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:16]
            if health.get("application") == "ecc-wouaf-local-ui" and health.get("version") == "3.0.4" and health.get("checkout_id") == checkout_id:
                if not args.no_browser:
                    webbrowser.open(url)
                print("LOCAL_UI_ALREADY_RUNNING", url, flush=True)
                return
        except Exception:
            pass
        server = LocalServer(("127.0.0.1", 0), args.device)
    record = {"url": server.url, "device": args.device, "distribution": "3.0.4",
              "pid": os.getpid(), "checkout_id": server.checkout_id,
              "started_at": datetime.now(timezone.utc).isoformat()}
    (ROOT / "outputs/ui_server.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("LOCAL_UI_READY", server.url, flush=True)
    if not args.no_browser:
        threading.Timer(.5, webbrowser.open, args=(server.url,)).start()
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("LOCAL_UI_STOPPED", flush=True)


if __name__ == "__main__":
    main()

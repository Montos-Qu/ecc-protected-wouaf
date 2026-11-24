"""Exercise real UI inference, original-byte routing and local HTTP boundaries."""
import io
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecc_wouaf import Predictor
from PIL import Image


class WebProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = (ROOT / "outputs/ui_test_server.log").open("w", encoding="utf-8")
        started = time.time()
        cls.process = subprocess.Popen([sys.executable, "-I", "-X", "utf8", str(ROOT / "wouaf.py"), "ui", "--port", "0", "--no-browser"], cwd=ROOT, stdout=cls.log, stderr=subprocess.STDOUT)
        cls.url = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            record = ROOT / "outputs/ui_server.json"
            if record.is_file():
                try:
                    value = json.loads(record.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    time.sleep(.1)
                    continue
                if datetime.fromisoformat(value["started_at"]).timestamp() < started:
                    time.sleep(.1)
                    continue
                url = value["url"]
                try:
                    with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                        health = json.load(response)
                    if health.get("version") == "3.0.4":
                        cls.url = url
                        break
                except (OSError, ValueError):
                    pass
            if cls.process.poll() is not None:
                raise RuntimeError("UI test server stopped during startup")
            time.sleep(.1)
        if not cls.url:
            cls.process.terminate()
            raise RuntimeError("UI test server did not become ready")
        with urllib.request.urlopen(cls.url, timeout=5) as response:
            page = response.read().decode("utf-8")
        cls.token = re.search(r'name="wouaf-token" content="([0-9a-f]+)"', page).group(1)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.request("/api/shutdown", b"")
            cls.process.wait(timeout=10)
        finally:
            if cls.process.poll() is None:
                cls.process.terminate()
            cls.log.close()

    @classmethod
    def request(cls, path, data=None, headers=None):
        values = {"Content-Type": "application/octet-stream", "X-WOUAF-Token": cls.token}
        values.update(headers or {})
        request = urllib.request.Request(cls.url + path, data=data, headers=values)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()

    def test_real_versions_match_predictor(self):
        images = ["demo/images/png8_control/positive_k11/10000000.png",
                  "demo/images/jpeg_q90_420/positive_k11/10000000.jpg",
                  "demo/images/png8_control/generated_negative/10000000.png"]
        for version, relative, code in [("v3", images[0], 11), ("v1", images[0], 11),
                                        ("v2", images[0], 11), ("v3", images[1], 11),
                                        ("v3", images[2], 11), ("v3", "demo/images/png8_control/positive_k16/10000000.png", 16)]:
            path = ROOT / relative
            data = path.read_bytes()
            expected = Predictor(version, "cpu", False).predict_encoded(data, code, path.name)
            status, payload = self.request("/api/predict?version=" + version + "&code=" + str(code), data, {"X-Filename": path.name})
            self.assertEqual(status, 200)
            response = json.loads(payload)
            self.assertEqual(response["result"], expected)
            status, downloaded = self.request(response["download_url"])
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(downloaded), expected)
            if relative == images[0] and version == "v3":
                self.assertEqual(expected["user_id"], 565)
            if relative == images[2]:
                self.assertEqual(expected["decision"], "REJECT")
                self.assertIsNone(expected["user_id"])

    def test_actual_header_controls_routing(self):
        data = (ROOT / "demo/images/jpeg_q90_420/positive_k11/10000000.jpg").read_bytes()
        status, payload = self.request("/api/predict", data, {"X-Filename": "renamed.png"})
        result = json.loads(payload)["result"]
        self.assertEqual(status, 200)
        self.assertEqual(result["selected_branch"], "jpeg420_expert")
        self.assertEqual(result["container"], "JPEG")

    def test_invalid_images_and_configuration(self):
        for path, data, expected in [("/api/predict", b"not an image", "invalid_image"),
                                     ("/api/predict?version=v9", b"x", "invalid_configuration"),
                                     ("/api/predict?code=21", b"x", "invalid_configuration")]:
            status, response = self.request(path, data)
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(response)["error"], expected)
        for size, kind, expected in [((32, 32), "PNG", "wrong_dimensions"), ((512, 512), "GIF", "unsupported_format")]:
            stream = io.BytesIO()
            Image.new("RGB", size).save(stream, format=kind)
            status, response = self.request("/api/predict", stream.getvalue())
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(response)["error"], expected)
        self.assertEqual(self.request("/api/predict", b"", {"Content-Length": str(10 * 1024 * 1024 + 1)})[0], 413)

    def test_loopback_and_session_boundaries(self):
        for headers in [{"Origin": "https://example.com"}, {"Host": "example.com"}, {"X-WOUAF-Token": ""}]:
            self.assertEqual(self.request("/api/predict", b"x", headers)[0], 403)
        self.assertEqual(self.request("/../models/v1_negative_aware_2400.pth")[0], 404)
        self.assertEqual(self.request("/api/sample?id=../../models")[0], 404)

    def test_optional_fit_and_noop_native_equivalence(self):
        from image_preprocessing import prepare_image
        path = ROOT / "demo/images/png8_control/positive_k11/10000000.png"
        original = path.read_bytes()
        expected = Predictor("v3", "cpu", False).predict_encoded(original, 11, path.name)
        status, body = self.request("/api/predict?resize=fit", original, {"X-Filename": path.name})
        response = json.loads(body)
        self.assertEqual(status, 200)
        actual = response["result"]
        self.assertFalse(actual.pop("preprocessing")["applied"])
        self.assertEqual(actual, expected)
        self.assertIsNone(response["processed_preview"])
        for version in ("v3", "v1", "v2"):
            with Image.open(io.BytesIO(original)) as image:
                buffer = io.BytesIO()
                image.resize((1024, 1024), Image.Resampling.LANCZOS).save(buffer, format="PNG")
            status, body = self.request("/api/predict?resize=fit&version=" + version, buffer.getvalue())
            result = json.loads(body)
            self.assertEqual(status, 200)
            self.assertEqual(result["result"]["requested_version"], version)
            self.assertEqual(result["result"]["preprocessing"]["target_size"], [512, 512])
            self.assertTrue(result["processed_preview"].startswith("data:image/png;base64,"))
        stream = io.BytesIO()
        Image.new("RGB", (1000, 500)).save(stream, format="PNG")
        prepared = prepare_image(stream.getvalue(), "fit")
        self.assertEqual(prepared.preprocessing["fitted_size"], [512, 256])
        self.assertEqual(prepared.preprocessing["padding_ltrb"], [0, 128, 0, 128])
        with self.assertRaisesRegex(ValueError, "wrong_dimensions"):
            prepare_image(stream.getvalue(), "native")
        stream = io.BytesIO()
        Image.new("RGB", (8193, 1)).save(stream, format="PNG")
        self.assertEqual(self.request("/api/predict?resize=fit", stream.getvalue())[0], 400)
        self.assertEqual(self.request("/api/predict?resize=stretch", original)[0], 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)

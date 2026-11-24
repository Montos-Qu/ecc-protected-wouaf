"""Measure optional fitting on existing labeled native images without training."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin
from ecc_wouaf import Predictor
from image_preprocessing import predict_encoded, prepare_image

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("native512", "auto512", "reencode512", "up1024", "down256", "wide768x512")


def transformed(data, scenario):
    if scenario in ("native512", "auto512"):
        return data
    with Image.open(io.BytesIO(data)) as image:
        container = image.format
        sampling = JpegImagePlugin.get_sampling(image) if container == "JPEG" else None
        pixels = image.convert("RGB")
    if scenario == "up1024":
        pixels = pixels.resize((1024, 1024), Image.Resampling.LANCZOS)
    elif scenario == "down256":
        pixels = pixels.resize((256, 256), Image.Resampling.LANCZOS)
    elif scenario == "wide768x512":
        canvas = Image.new("RGB", (768, 512), (128, 128, 128))
        canvas.paste(pixels, (128, 0))
        pixels = canvas
    output = io.BytesIO()
    if container == "JPEG":
        pixels.save(output, format="JPEG", quality=90, subsampling=sampling, optimize=False, progressive=False)
    else:
        pixels.save(output, format="PNG")
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Choose a new output directory; previous resize results are never overwritten")
    args.out.mkdir(parents=True)
    manifest_path = ROOT / "demo/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = {"study": "optional_resize_existing_real_demo_v1", "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "native_encoded_files": len(manifest), "unique_source_images": len({(x["group"], x["sample_id"]) for x in manifest}),
            "scenarios": list(SCENARIOS), "models": ["v3"], "codes": [11, 16], "device": args.device,
            "weights_and_gates_fixed": True, "new_training_steps": 0,
            "encoding_protocol": "Keep input PNG/JPEG container. Derived JPEG uses Pillow Q90, original subsampling, optimize=False, progressive=False. Auto fit uses LANCZOS, centered RGB128 padding and a lossless PNG handoff; routing retains the incoming container header.",
            "interpretation": "Existing regression fixtures, not a fresh independent cohort. Correlated views are not independent images. Four positive source images per code. Report rates descriptively; do not claim population robustness.",
            "software_release_checks": ["native and auto512 complete results equal after removing preprocessing metadata", "uploaded source hashes match manifest", "actual input header routing preserved", "correct+reject+false equals sample count", "no training or threshold changes"],
            "quality_policy": "Non-native geometry remains explicitly experimental regardless of these small-sample rates; native mode stays default. Reencode512 separates an additional JPEG encoding from geometry."}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    model = Predictor("v3", args.device, False)
    baselines = {}
    rows = []
    identical = 0
    predictions = (args.out / "predictions.jsonl").open("w", encoding="utf-8")
    try:
        for scenario in SCENARIOS:
            for k in (11, 16):
                subset = [x for x in manifest if x["group"] in ("positive_k" + str(k), "generated_negative", "natural_negative")]
                for item in subset:
                    raw = (ROOT / item["image"]).read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == item["sha256"], item["image"]
                    data = transformed(raw, scenario)
                    mode = "native" if scenario == "native512" else "fit"
                    result = predict_encoded(model, data, k, item["image_id"], mode)
                    key = (k, item["image_id"])
                    if scenario == "native512":
                        baselines[key] = result
                    if scenario == "auto512":
                        stripped = dict(result)
                        stripped.pop("preprocessing")
                        assert stripped == baselines[key]
                        assert result["preprocessing"]["original_bytes_preserved"]
                        identical += 1
                    expected_branch = "jpeg420_expert" if item["condition"] == "jpeg_q90_420" else "clean_v1"
                    assert result["selected_branch"] == expected_branch
                    positive = item["group"].startswith("positive")
                    correct = positive and result["decision"] == "ACCEPT" and result["user_id"] == item["user_id"]
                    false = result["decision"] == "ACCEPT" and not correct
                    reject = result["decision"] == "REJECT"
                    assert int(correct) + int(false) + int(reject) == 1
                    error_count = int(np.count_nonzero(np.asarray(result["pred_bits"]) != item["gt_bits"])) if positive else None
                    row = {"scenario": scenario, "code": k, "condition": item["condition"], "group": item["group"],
                           "sample_id": item["sample_id"], "scene_id": item["scene_id"], "image_id": item["image_id"],
                           "gt_user_id": item["user_id"], "gt_bits": item["gt_bits"], "transformed_input_sha256": hashlib.sha256(data).hexdigest(),
                           "correct": bool(correct), "reject": reject, "false_attribution": bool(false),
                           "Hamming_error_count": error_count, "result": result}
                    predictions.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows.append(row)
                print("RESIZE_REGRESSION", scenario, k, len(rows), flush=True)
    finally:
        predictions.close()
    summary = []
    for scenario in SCENARIOS:
        for k in (11, 16):
            for condition in ("png8_control", "jpeg_q90_444", "jpeg_q90_420"):
                for group in ("positive_k" + str(k), "generated_negative", "natural_negative"):
                    cohort = [r for r in rows if (r["scenario"], r["code"], r["condition"], r["group"]) == (scenario, k, condition, group)]
                    n = len(cohort)
                    counts = {field: sum(r[field] for r in cohort) for field in ("correct", "reject", "false_attribution")}
                    assert sum(counts.values()) == n
                    summary.append({"scenario": scenario, "code": k, "condition": condition, "group": group,
                                    "images": n, **counts, "correct_rate": counts["correct"] / n,
                                    "reject_rate": counts["reject"] / n, "false_attribution_rate": counts["false_attribution"] / n,
                                    "BER": sum(r["Hamming_error_count"] for r in cohort) / (32 * n) if group.startswith("positive") else ""})
    with (args.out / "recovery_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    aggregate = []
    for scenario in SCENARIOS:
        for k in (11, 16):
            selected = [r for r in rows if r["scenario"] == scenario and r["code"] == k]
            positive = [r for r in selected if r["gt_bits"] is not None]
            aggregate.append({"scenario": scenario, "code": k, "positive_views": len(positive), "negative_views": len(selected) - len(positive),
                              "correct_positive": sum(r["correct"] for r in positive), "rejected_positive": sum(r["reject"] for r in positive),
                              "false_positive_identity": sum(r["false_attribution"] for r in positive),
                              "accepted_negative": sum(r["false_attribution"] for r in selected if r["gt_bits"] is None)})
    report = {"complete": True, "software_checks_passed": True, "auto512_exact_native_result_matches": identical,
              "prediction_records": len(rows), "source_encoded_files": len(manifest), "unique_source_images": plan["unique_source_images"],
              "new_training_steps": 0, "thresholds_changed": False, "weights_changed": False,
              "experimental_geometry_only": True, "fresh_independent_test": False, "aggregate_by_code_and_scenario": aggregate}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

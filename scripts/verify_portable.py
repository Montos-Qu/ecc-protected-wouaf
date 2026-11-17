"""Exercise unchanged trained V1/V2/V3 on native fixtures and report runtime differences."""
import argparse
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from ecc_wouaf import Predictor
from ecc_wouaf.core import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/portable_verification.json")
    args = parser.parse_args()
    golden = json.loads((ROOT / "demo/golden_predictions.json").read_text(encoding="utf-8"))
    systems = {version: Predictor(version, args.device, strict=False) for version in ("v1", "v2", "v3")}
    changed_bits = 0
    changed_decisions = []
    max_logit_difference = 0.0
    fixed_controls = 0
    for index, row in enumerate(golden):
        path = ROOT / row["image"]
        assert sha256(path) == row["sha256"]
        expected = row["reference"]["v3"]
        k = int(next(iter(expected["decisions"])))
        actual = systems["v3"].predict(path, k)
        assert actual["model_status"] == "trained_final_research_system"
        assert len(actual["logits"]) == 32 and np.isfinite(actual["logits"]).all()
        assert actual["selected_branch"] == expected["selected_branch"]
        delta = float(np.max(np.abs(np.asarray(actual["logits"]) - np.asarray(expected["logits"]))))
        max_logit_difference = max(max_logit_difference, delta)
        changed_bits += int(np.sum(np.asarray(actual["pred_bits"]) != np.asarray(expected["pred_bits"])))
        model = systems["v3"].model(actual["selected_branch"])
        for code, reference in expected["decisions"].items():
            decoded = model.decoder.decode(np.asarray([actual["logits"]], np.float32),
                                           np.asarray([actual["presence_logit"]], np.float32), int(code))
            accepted = bool(decoded["accepted"][0])
            candidate = int(decoded["candidate_user_id"][0])
            if accepted != reference["accepted"] or accepted and candidate != reference["candidate_user_id"]:
                changed_decisions.append({"image": row["image"], "k": int(code)})
        fixed = "v2" if actual["selected_branch"] == "jpeg420_expert" else "v1"
        control = systems[fixed].predict(path, k)
        for field in ("logits", "pred_bits", "decision", "user_id"):
            assert control[field] == actual[field], (row["image"], fixed, field)
        fixed_controls += 1
        if actual["decision"] == "REJECT":
            assert actual["user_id"] is None
        if (index + 1) % 8 == 0:
            print("PORTABLE_FIXTURES", index + 1, len(golden), flush=True)
    jpeg = next(row for row in golden if row["condition"] == "jpeg_q90_420")
    renamed = systems["v3"].predict_encoded((ROOT / jpeg["image"]).read_bytes(), 11, "renamed.png")
    assert renamed["selected_branch"] == "jpeg420_expert"
    small = io.BytesIO()
    Image.new("RGB", (16, 16)).save(small, format="PNG")
    try:
        systems["v3"].predict_encoded(small.getvalue())
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported input size was accepted")
    passed = not changed_decisions
    report = {"passed": passed, "runtime": "portable", "reference_runtime_verified": False,
              "system": platform.system(), "python": platform.python_version(), "device": args.device,
              "native_fixture_files": len(golden), "V1_V2_branch_equivalence_controls": fixed_controls,
              "maximum_A100_reference_logit_difference": max_logit_difference,
              "bit_differences_from_A100_reference": changed_bits,
              "changed_accepted_attribution_decisions": changed_decisions,
              "weights_and_thresholds_changed": False, "new_training_steps": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    receipt_path = ROOT / "outputs/portable_install.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["model_validation_passed"] = passed
    receipt["validation_report"] = str(args.output)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit("Portable fixture decisions differ from the saved reference; review the report before use.")


if __name__ == "__main__":
    main()

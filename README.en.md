# ECC-Protected WOUAF — pretrained final V3

The default entry loads the trained final research system. V3 routes actual JPEG 4:2:0 files to the V2 expert and native PNG/JPEG444 files to V1. It contains two immutable pretrained extractors and performs one network forward per image. It is not a newly trained single network. V1 and V2 remain selectable controls. Training lives separately in `training/` and never runs during inference.

See the detailed [Chinese README](README.md) for the complete workflow, package inventory, frozen environment, training recipes, reproduction commands, results, and limitations.

## Quick start

Clone this repository and run `python scripts/download_pretrained.py`, or download the complete pretrained ZIP from the v3.0.0 GitHub Release. The two real checkpoint files are distributed through Release assets, with their original SHA256 values. No training is required.

The frozen environment, generator, and full image/latent archives remain in the [Drive release directory](https://drive.google.com/drive/u/0/folders/1CTqCSbCHS480Hz6xRbp17KvvmTad3_cq) under `offline_assets/`, subject to its existing access permissions. Obtain/download/mount those assets and point `ASSETS` to that directory before strict reproduction. They are separate from both the Git checkout and the main project ZIP.

```bash
python scripts/download_pretrained.py
python scripts/materialize_assets.py --asset-root "$ASSETS" --id frozen_environment
python scripts/restore_environment.py
python scripts/run_locked.py --check
python scripts/verify_files.py
python scripts/run_locked.py scripts/verify_pretrained.py --output outputs/demo_verification.json
python scripts/run_locked.py infer.py --image demo/images/png8_control/positive_k11/10000000.png --output outputs/prediction.json
```

Omit `--version` for V3; use `--version v1` or `--version v2` for controls. Primary BCH(31,11)+overall parity supports 2048 users; `--code 16` selects the 65536-user capacity control. A rejection returns `user_id: null`. Outputs retain actual format/sampling, selected branch, model hash, bits, logits, probabilities, presence score, and rejection reason. The CLI reads existing encoded files without recompression or silent resizing.

## Independent-content validation

4096 source images, 12288 actual encoded views, 1024 positive images per code and condition. Fresh prompts/seeds/latents and natural-photo IDs/hashes were locked after the models and rules were frozen. The primary endpoint was k11 JPEG420 accepted exact-user recovery.

| Code | Native PNG V1→V3 | JPEG444 Q90 V1→V3 | JPEG420 Q90 V1→V3 |
|---|---:|---:|---:|
| k11 | 90.14%→90.14% | 87.60%→87.60% | **74.12%→76.86%** |
| k16 | 67.68%→67.68% | 56.35%→56.35% | **33.40%→37.40%** |

Paired 64-scene-cluster bootstrap, 10000 resamples: JPEG420 gain +2.73 pp [1.37,4.10] for k11; +4.00 pp [2.64,5.47] for k16. PNG/444 saved outputs match per sample. Zero false attributions were observed in this finite study; this is not a zero population-risk guarantee. Historical k11 evaluation identities may repeat; independence concerns fresh content and recorded project data, not upstream foundation-model pretraining.

## Reproduction

`analysis/recompute.py --out NEW_DIRECTORY` re-decodes saved predictions and recomputes all full-cohort statistics without network inference. `evaluate.py` reads restored native files, defaults to V3, and exports new per-image predictions. The complete image/latent assets are stored separately with checksums. `generation/from_latents.py` and `generation/sample_latents.py` explicitly reconstruct the frozen generator workflow. The clean notebook provides the default pretrained demo.

Use the archived Python3.10.21 / torch1.12.1+cu116 environment and reference A100/driver profile; do not upgrade packages. Re-execution on other software/hardware is an explicitly unvalidated mode. Portable training entries were dry-run checked; the release packaging does not claim a new full-training rerun or byte-identical retraining weights.

The qualification covers 512×512 native PNG and Pillow JPEG Q90 444/420. Converting previously compressed JPEG pixels into a real PNG erases sampling provenance; the router then uses V1. Other sizes, sampling modes, codecs, and attacks need separate evaluation. Third-party notices remain in `THIRD_PARTY_NOTICES.md` and `licenses/`. No new blanket license is applied to third-party models, images, or environment components. The original Drive archive is immutable; GitHub packaging changes are recorded in `reports/github_publication.json`.

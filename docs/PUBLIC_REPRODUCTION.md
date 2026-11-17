[English](#english) | [中文](#chinese)

<a name="english"></a>

# Public downloads and reproduction

## Use the trained product locally

On Windows, extract the complete Release ZIP, run `setup_windows.cmd`, then drop an image onto `predict_windows.cmd`. No preinstalled Python is required. With a host Python3.9+ on Windows x64 or Linux x86_64:

```bash
python wouaf.py setup
python wouaf.py predict --image YOUR_IMAGE.png
```

Setup obtains the original V1/V2 weights and support data when missing, installs an isolated Python3.12.12/PyTorch2.7.1 runtime from locked official binaries, and runs model checks. V3 is default. Add `--version v1` or `--version v2` to predict for a control. Normal setup never downloads the old A100 environment and never trains. CPU is available without NVIDIA; usable supported NVIDIA GPUs can use CUDA. See [portable runtime](PORTABLE_RUNTIME.md) for hardware choices and actual validation scope.

| Goal | Command | Additional assets |
|---|---|---|
| Trained V3/V1/V2 and small self-check | `python wouaf.py setup` | Weights/support if missing; managed local runtime |
| Full independent-image evaluation | `python wouaf.py setup --with-evaluation` | Six native image groups, about2.34GiB compressed |
| Statistics from saved predictions | `python scripts/run_portable.py analysis/recompute.py --help` | Support predictions, no full images or network forwards |
| Strict historical reference | `python wouaf.py setup --runtime reference` | Original2.16GiB Linux environment archive; historical A100/driver |
| Regenerate/retrain | Separate `generation/` and `training/` | Optional research assets outside ordinary inference |

## Evaluate existing native images

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
python scripts/run_portable.py analysis/recompute.py --v1 outputs/independent_v1 --v3 outputs/independent_v3 --out outputs/recomputed_independent
```

The cohort contains4096 source images/12288 encoded views with saved GT, prompts and manifests. Evaluation reads native PNG/JPEG files, without regeneration, resizing or recompression. Default k11 and capacity control k16 use frozen thresholds. Portable results must retain their runtime identity; they are not automatically strict reference results.

## Public asset links

All links below are read-only public. Ordinary inference does not need a Google login or private parent-folder access. The original frozen archive is optional, for reference mode only. The exact sizes and SHA256 hashes are also in [public_assets.json](../public_assets.json).

| Public asset group | Download link | Download size |
|---|---|---:|
| Historical reference environment (three parts) | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| k11 positive PNGs | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| k16 positive PNGs | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |
| Generated negative PNGs | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| Natural negative PNGs | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| Native JPEG444 Q90 views | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| Native JPEG420 Q90 views | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |

| File | File link | Exact bytes | SHA256 |
|---|---|---:|---|
| `frozen_environment/frozen_environment.part000` | [Download](https://drive.google.com/file/d/1sMErk4a_C8ZK6w3i5EauY1Lz0Rqjo2iD/view) | 1073741824 | `3ff86ba424249644f44b2cbd1e26a3b35bc51f4cb03aaf46ee3407e93e6a368c` |
| `frozen_environment/frozen_environment.part001` | [Download](https://drive.google.com/file/d/1NfI0n6RExdjsnao_1X8DBYdf6Z9Zi9b_/view) | 1073741824 | `aec5f627b697ce4c9a9bec4f5276bde5b74fe5b74c69cb7695738702d4a3d232` |
| `frozen_environment/frozen_environment.part002` | [Download](https://drive.google.com/file/d/1HNy9xBzNpYTaYWj4w3qsYnD9sdXNZTKo/view) | 172268060 | `2504c964cd9a68c3df896180d460233593f859f764ba0094c824e35ff7a208b9` |
| `independent_source_pngs_positive_k11/source_pngs_positive_k11.zip` | [Download](https://drive.google.com/file/d/1z5PhK0Nzh3c9Blnshn_C7CZC9PsYJ1cw/view) | 458231358 | `5b19e774d6752202184386e7792ce0a19865ebe33b7f209be90a9958da63a7cd` |
| `independent_source_pngs_positive_k16/source_pngs_positive_k16.zip` | [Download](https://drive.google.com/file/d/17I4z-4WXCXmxXWOwOZDP8jbFAZQidwMU/view) | 458263430 | `31a1356342449bf1cae4815ec73a7d35dd32630adcedc4fedba45eea3e087e3e` |
| `independent_source_pngs_generated_negative/source_pngs_generated_negative.zip` | [Download](https://drive.google.com/file/d/14hQBofKtDUBQWSbzNwudQMlyHdNA-1qt/view) | 472933398 | `c8c82b0f41532089f909aacaa6e39eea376eebe435d87158317c45e822ae6fb4` |
| `independent_source_pngs_natural_negative/source_pngs_natural_negative.zip` | [Download](https://drive.google.com/file/d/1ybyhh1xrULcDJ6pnMyvNweFm8jRHHii9/view) | 409355808 | `bf62e115490a2b64b213b62a895660c8e253962a7a4b02f1a25abf7b8c0d3cb2` |
| `independent_encoded_jpeg_q90_444/encoded_jpeg_q90_444.zip` | [Download](https://drive.google.com/file/d/1ZlUaxakbTBsyVOtmKpP-c5nEqZTIjqSe/view) | 396389848 | `9bb9f9b535c5f6ed3e29e631739176c49646eeb52fc38a56ed525e9267fb1dac` |
| `independent_encoded_jpeg_q90_420/encoded_jpeg_q90_420.zip` | [Download](https://drive.google.com/file/d/1sNMYit3B050-lx2GZL_vHpKKYFZst09G/view) | 315041629 | `e85aa98b9322efcd5a039b602196c13a4615a16b5c7e26089c8add34fb84b01f` |

## Optional historical reference mode

```bash
python wouaf.py setup --runtime reference
python wouaf.py predict --runtime reference --image YOUR_IMAGE.png
```

This restores the original Linux Python3.10.21 environment to `/content/strict_wouaf_restore/env` and preserves its91 Python distributions,23 native builds, A100-SXM4-80GB/driver580.82.07 checks. Linux host Python3.12+ is used only for bootstrap. Add `--with-evaluation` for the images and use `--runtime reference` on each evaluate command. Reference statistics use `scripts/run_locked.py` instead of `scripts/run_portable.py`. See [environment details](ENVIRONMENT.md).

The public downloader uses anonymous HTTP, handles Drive download confirmation and verifies sizes/SHA256. Existing valid files are reused; conflicting files fail without overwrite. Optional downloads live in `offline_assets/`, or `--asset-root PATH`. Drive quotas may require retrying later or manually placing a linked file at the documented path. Model artifacts, codebooks and experiment objects remain the original frozen version; product patch releases update distribution only. Public links do not change upstream licenses or grant access to other private training/generator backups.

---

<a name="chinese"></a>

# 公开下载与复现

## 在个人电脑使用成品

Windows用户解压完整Release ZIP，运行 `setup_windows.cmd`，随后把图片拖到 `predict_windows.cmd`；不必预装Python。已有Python3.9+的Windows x64或Linux x86_64用户执行：

```bash
python wouaf.py setup
python wouaf.py predict --image YOUR_IMAGE.png
```

安装器在需要时获取原V1/V2权重及支撑数据，从锁定的官方安装包创建独立Python3.12.12/PyTorch2.7.1环境，并运行模型自检。默认V3，预测加 `--version v1` 或 `--version v2` 可选对照。普通安装不下载旧A100环境，也不训练。没有NVIDIA也能用CPU，受支持且驱动可用的NVIDIA显卡可用CUDA。硬件选择及实测范围见[普通模式说明](PORTABLE_RUNTIME.md)。

| 目标 | 命令 | 额外资产 |
|---|---|---|
| 已训练V3/V1/V2及小量自检 | `python wouaf.py setup` | 缺失的权重/支撑文件、独立本地环境 |
| 完整独立图片验证 | `python wouaf.py setup --with-evaluation` | 六组原生图片，压缩后约2.34GiB |
| 用保存预测重算统计 | `python scripts/run_portable.py analysis/recompute.py --help` | 支撑预测，无须完整图片或模型前向 |
| 严格历史参考模式 | `python wouaf.py setup --runtime reference` | 原2.16GiB Linux环境，历史A100/驱动 |
| 再生成/训练 | 独立 `generation/` 和 `training/` | 普通推理之外的可选研究资产 |

## 评估现有原生图片

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
python scripts/run_portable.py analysis/recompute.py --v1 outputs/independent_v1 --v3 outputs/independent_v3 --out outputs/recomputed_independent
```

队列含4096源图/12288编码视图，GT、prompt和manifest已保存。评估读取原生PNG/JPEG，不重新生成、缩放或压缩。默认k11及容量对照k16使用冻结门槛。普通模式的新结果应保留运行环境标记，不能自动当成严格历史复现结果。

## 公开资产链接

下列链接均为公开只读。普通推理无需Google登录或私有父目录权限。原冻结环境只用于可选参考模式。精确字节数及SHA256也保存在 [public_assets.json](../public_assets.json)。

| 公开文件组 | 下载链接 | 下载大小 |
|---|---|---:|
| 冻结 Python、PyTorch 与全部依赖（3 个分片） | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| k11 水印 PNG 正样本 | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| k16 水印 PNG 正样本 | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |
| 同源无水印 PNG 负样本 | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| 自然图 PNG 负样本 | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| 真实 JPEG444 Q90 验证视图 | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| 真实 JPEG420 Q90 验证视图 | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |

| 文件 | 文件入口 | 精确字节数 | SHA256 |
|---|---|---:|---|
| `frozen_environment/frozen_environment.part000` | [下载](https://drive.google.com/file/d/1sMErk4a_C8ZK6w3i5EauY1Lz0Rqjo2iD/view) | 1073741824 | `3ff86ba424249644f44b2cbd1e26a3b35bc51f4cb03aaf46ee3407e93e6a368c` |
| `frozen_environment/frozen_environment.part001` | [下载](https://drive.google.com/file/d/1NfI0n6RExdjsnao_1X8DBYdf6Z9Zi9b_/view) | 1073741824 | `aec5f627b697ce4c9a9bec4f5276bde5b74fe5b74c69cb7695738702d4a3d232` |
| `frozen_environment/frozen_environment.part002` | [下载](https://drive.google.com/file/d/1HNy9xBzNpYTaYWj4w3qsYnD9sdXNZTKo/view) | 172268060 | `2504c964cd9a68c3df896180d460233593f859f764ba0094c824e35ff7a208b9` |
| `independent_source_pngs_positive_k11/source_pngs_positive_k11.zip` | [下载](https://drive.google.com/file/d/1z5PhK0Nzh3c9Blnshn_C7CZC9PsYJ1cw/view) | 458231358 | `5b19e774d6752202184386e7792ce0a19865ebe33b7f209be90a9958da63a7cd` |
| `independent_source_pngs_positive_k16/source_pngs_positive_k16.zip` | [下载](https://drive.google.com/file/d/17I4z-4WXCXmxXWOwOZDP8jbFAZQidwMU/view) | 458263430 | `31a1356342449bf1cae4815ec73a7d35dd32630adcedc4fedba45eea3e087e3e` |
| `independent_source_pngs_generated_negative/source_pngs_generated_negative.zip` | [下载](https://drive.google.com/file/d/14hQBofKtDUBQWSbzNwudQMlyHdNA-1qt/view) | 472933398 | `c8c82b0f41532089f909aacaa6e39eea376eebe435d87158317c45e822ae6fb4` |
| `independent_source_pngs_natural_negative/source_pngs_natural_negative.zip` | [下载](https://drive.google.com/file/d/1ybyhh1xrULcDJ6pnMyvNweFm8jRHHii9/view) | 409355808 | `bf62e115490a2b64b213b62a895660c8e253962a7a4b02f1a25abf7b8c0d3cb2` |
| `independent_encoded_jpeg_q90_444/encoded_jpeg_q90_444.zip` | [下载](https://drive.google.com/file/d/1ZlUaxakbTBsyVOtmKpP-c5nEqZTIjqSe/view) | 396389848 | `9bb9f9b535c5f6ed3e29e631739176c49646eeb52fc38a56ed525e9267fb1dac` |
| `independent_encoded_jpeg_q90_420/encoded_jpeg_q90_420.zip` | [下载](https://drive.google.com/file/d/1sNMYit3B050-lx2GZL_vHpKKYFZst09G/view) | 315041629 | `e85aa98b9322efcd5a039b602196c13a4615a16b5c7e26089c8add34fb84b01f` |

## 可选历史参考模式

```bash
python wouaf.py setup --runtime reference
python wouaf.py predict --runtime reference --image YOUR_IMAGE.png
```

恢复原Linux Python3.10.21环境到 `/content/strict_wouaf_restore/env`，保留91个Python distributions、23个native builds、A100-SXM4-80GB/driver580.82.07检查。Linux宿主Python3.12+仅负责启动恢复。追加 `--with-evaluation` 下载图片，每次evaluate也要明确 `--runtime reference`；重算统计时把 `scripts/run_portable.py` 改为 `scripts/run_locked.py`。见[环境说明](ENVIRONMENT.md)。

公开下载器使用匿名HTTP，处理Drive下载确认，并验证大小/SHA256。已有正确文件复用，不符文件拒绝覆盖。可选下载保存在 `offline_assets/`，也可用 `--asset-root PATH`。Drive配额受限时稍后重试，或从链接下载到指定路径。模型、码本和实验对象仍为原冻结版本，产品补丁只更新分发方式。公开链接不会改变第三方许可，也不开放其他私有训练/生成器备份。

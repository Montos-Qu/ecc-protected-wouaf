# Frozen baseline-3000 generator reproduction

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

This folder is for researchers who want to generate the original watermarked test images. You can skip it when using the app to read an existing image. The saved generator and latent files are optional research downloads, not files needed by normal setup. The details below explain how to replay those older experiments.

### Experiment details

Default pretrained attribution does not load the Stable Diffusion generator. The separate `baseline_generator` research asset includes SD2 UNet/text encoder/tokenizer/original PNDM scheduler, VAE, checkpoint3000, mapping network, original training source/commit/diff/environment records and the pristine VAE. Restore optional assets with:

```bash
python scripts/materialize_assets.py --asset-root "$ASSETS" --id baseline_generator independent_new_latents
```

`from_latents.py` restores custom VAE/mapping sources and renders with the original checkpoint and 32-bit codewords. It reads archived models, uses float32, disables TF32, preserves cudnn deterministic=False, latent scale .18215, mapping input bits+1e-8 and the original uint8 rounding. The independent cohort used batch4; this entry retains that batch size and exports PNGs with GT/latent/file hashes. Check the root research guide's limit4 command before explicitly requesting the full run.

`sample_latents.py` resamples from saved new prompts/seeds using local SD2 components, PNDM50, CFG7.5, per-unit seeded CPU initial noise and batch8. It does not automatically download new models or upgrade dependencies. Sampling all1024 units takes longer than restoring saved latents and is not part of the default demo.

For the exact verified JPEG bytes, restore `independent_encoded_jpeg_q90_444/420` directly. `encode_views.py` explicitly creates PNG/JPEG Q90 444/420 view manifests from PNG sources for historical regression or custom labeled data; default inference reads existing files without this reencoding.

`validation_latents` preserves the original5000 WOUAF latent tar.gz with its recorded directory structure, separately from the final1024 fresh latents. Original COCO natural photos are in `independent_natural_originals`, with cropped PNGs stored separately to preserve playback against future download/codec changes.

Original upstream full-training source and the3000-step checkpoint are research references under the restored asset's `code/` and `checkpoint-3000/`. The release verified portable extractor training and frozen generator restoration; it does not claim one-command exact replay of every upstream training branch. Optional generator/training/latent assets retain their separate access and licensing terms; they are not required for the pretrained product or public fixed-image evaluation.

---

<a name="chinese"></a>

## 中文

这里用于重新生成带指纹的实验图片。普通用户只想上传已有图片识别时，可以跳过整个目录。生成器和latent（生成时使用的中间数据）属于可选研究下载，不是安装成品的必需文件。下面保留重做旧实验的具体参数。

### 实验详情

默认归属不需要加载SD生成器。完整生成资产作为 `baseline_generator` 单独保存，包括SD2的UNet/text encoder/tokenizer/原PNDMscheduler、VAE、checkpoint3000、mapping、原训练脚本/commit/diff/环境记录及pristine VAE。恢复：

```bash
python scripts/materialize_assets.py --asset-root "$ASSETS" --id baseline_generator independent_new_latents
```

`from_latents.py`恢复自定义VAE/mapping源码，用原checkpoint与32-bit码字渲染；代码只读归档模型，float32，TF32关闭，cudnn deterministic=False，latent缩放.18215，mapping输入bits+1e-8，按原公式四舍五入uint8。原独立队列每批4张，此入口保持同一batch；输出PNG与逐图GT/latent/file hash。先用README的limit4命令核对，再明确去掉limit跑全量。

`sample_latents.py`从保存的新prompt/seed清单重采样，使用本地SD2组件、PNDM50steps、CFG7.5、CPU按单元seed生成初始噪声、batch8。默认没有联网下载模型或升级依赖。完整1024单元采样耗时明显长于直接恢复已存latent，不是默认demo的步骤。

如需保持已验证原生JPEG字节，直接恢复 `independent_encoded_jpeg_q90_444/420`。`encode_views.py`可从PNG源显式生成PNG/JPEGQ90 444/420评估清单，用于旧回归源图或用户自己的带标签数据；这与默认infer只读原文件不同。

`validation_latents`保留原5000个WOUAF latent tar.gz，按自身目录结构解压。它们与最终1024新latent分开。原COCO自然图在独立资产 `independent_natural_originals`；裁剪PNG另存，避免未来下载/编码库变化妨碍重放。

完整上游WOUAF从头训练代码和3000步断点可查恢复资产 `code/`、`checkpoint-3000/`；它们属于原训练来源和研究断点。本发布验证了提取器入口与固定生成器恢复范围，没有将所有上游全训练分支声称为可一键严格复现。

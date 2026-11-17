# 固定baseline-3000生成器复现

默认归属不需要加载SD生成器。完整生成资产作为 `baseline_generator` 单独保存，包括SD2的UNet/text encoder/tokenizer/原PNDMscheduler、VAE、checkpoint3000、mapping、原训练脚本/commit/diff/环境记录及pristine VAE。恢复：

```bash
python scripts/materialize_assets.py --asset-root "$ASSETS" --id baseline_generator independent_new_latents
```

`from_latents.py`恢复自定义VAE/mapping源码，用原checkpoint与32-bit码字渲染；代码只读归档模型，float32，TF32关闭，cudnn deterministic=False，latent缩放.18215，mapping输入bits+1e-8，按原公式四舍五入uint8。原独立队列每批4张，此入口保持同一batch；输出PNG与逐图GT/latent/file hash。先用README的limit4命令核对，再明确去掉limit跑全量。

`sample_latents.py`从保存的新prompt/seed清单重采样，使用本地SD2组件、PNDM50steps、CFG7.5、CPU按单元seed生成初始噪声、batch8。默认没有联网下载模型或升级依赖。完整1024单元采样耗时明显长于直接恢复已存latent，不是默认demo的步骤。

如需保持已验证原生JPEG字节，直接恢复 `independent_encoded_jpeg_q90_444/420`。`encode_views.py`可从PNG源显式生成PNG/JPEGQ90 444/420评估清单，用于旧回归源图或用户自己的带标签数据；这与默认infer只读原文件不同。

`validation_latents`保留原5000个WOUAF latent tar.gz，按自身目录结构解压。它们与最终1024新latent分开。原COCO自然图在独立资产 `independent_natural_originals`；裁剪PNG另存，避免未来下载/编码库变化妨碍重放。

完整上游WOUAF从头训练代码和3000步断点可查恢复资产 `code/`、`checkpoint-3000/`；它们属于原训练来源和研究断点。本发布验证了提取器入口与固定生成器恢复范围，没有将所有上游全训练分支声称为可一键严格复现。

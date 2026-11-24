[English](#english) | [中文](#chinese)

<a name="english"></a>

# Archived optional-fitting experiment (3.0.4)

**Removed from the current product in3.0.5.** This document describes only the historical3.0.4 experiment and its measurements. Fitting is no longer available in the current UI, CLI or API. Historical reproduction requires the immutable [v3.0.4 source](https://github.com/Montos-Qu/ecc-protected-wouaf/tree/v3.0.4). Current users should follow the [native image size policy](IMAGE_SIZE_LIMITS.md).

**Native512 remains the default. No training, weight or threshold changes.** In the local UI select **Image sizing → Fit to512 + padding · Experimental** to accept other sizes. The processed preview shows the exact512x512 model input. The original upload stays unmodified. CLI:

```bash
python wouaf.py predict --image YOUR_IMAGE.png
python wouaf.py predict --image LARGE_IMAGE.png --resize fit
```

Fit mode scales RGB pixels with Pillow LANCZOS while preserving aspect ratio, centers them on a512x512 RGB128 canvas when needed, and never crops or stretches. Existing512x512 images bypass all conversion and retain original encoded bytes. The incoming file header selects the same V3 branch before any processing; a resized JPEG420 still uses V2. The internal processed handoff is lossless PNG, with no extra JPEG compression by the feature. JSON records source/target/fitted dimensions, padding, method, source and processed hashes, and experimental scope. No EXIF autorotation is added. The original input-container/sampling fields describe the upload, not the intermediate PNG.

The existing10MiB upload cap remains. Auto fit allows at most16,777,216 pixels total and8192 per side. Native mode still rejects non512 inputs. Strict historical reference mode stays native-only; fitting is available through the portable profile.

## Actual labeled-image regression

The existing fixture set contains48 encoded files from16 source images, including PNG/JPEG444/JPEG420 and both codes' positive samples, generated negatives and natural negatives. Each code has4 positive source images, each in3 formats (12 correlated positive views), plus8 negative sources in3 formats (24 negative views) per transformation. These are compatibility fixtures from prior experiments, **not a fresh independent cohort or432 independent images**.

Six scenarios ×2 codes ×36 views =432 real inference records. All72 auto512 results matched native results exactly after removing the added preprocessing metadata; input bytes were retained. Five real HTTP test methods and browser native rejection, auto fitting, aspect-preserving padding, processed preview, language switching and result download passed.

| Transformation | k11 correct /12 positive views | k16 correct /12 positive views |
|---|---:|---:|
| Original512 | 8/12 | 5/12 |
| Auto fit on original512 | 8/12 | 5/12 |
| One reencoding control | 8/12 | 5/12 |
| 512 →1024 →auto512 | 4/12 | 2/12 |
| 512 →256 →auto512 | 0/12 | 0/12 |
| Centered margins →768x512 →auto fit | 0/12 | 0/12 |


No wrong accepted identity or accepted negative was observed in these432 records. Counts are descriptive, highly correlated and small; they do not establish zero false-attribution risk or a population recovery rate. Rejected positives account for the remaining positive views. Original large-cohort research rates stay unchanged and must not be replaced by these fixture counts.

The one-reencoding control keeps size/container, with JPEG Q90 and original sampling. Other derived inputs retain PNG/JPEG containers; JPEG is reencoded at Q90 before upload. This extra encoding is separated by the control, but its interactions with scaling cannot be assumed away. The wide-image test adds gray margins around an existing square source before fitting; it does not validate arbitrary non-square photographs. Full protocols, original GT, per-image logits/probabilities/bits, rejection reasons, Hamming errors and hashes are separately saved in [plan](../reports/resize_v3_0_4/plan.json), [summary](../reports/resize_v3_0_4/summary.json), [cells](../reports/resize_v3_0_4/recovery_cells.csv), and [predictions](../reports/resize_v3_0_4/predictions.jsonl).

**Interpretation:** fitting enables ingestion of other sizes; measured recovery worsened. Downsampled and padded fixtures were all rejected. This release passes software/native-preservation checks, not a claim of successful geometric robustness. Auto fit remains opt-in and experimental; use the original512 image when available. No result was used to retune a threshold or train a model.

Reproduce on the supplied images (use a new output directory; old results are never overwritten):

```bash
python scripts/run_portable.py analysis/resize_regression.py --out outputs/resize_my_run
python scripts/run_portable.py tests/test_local_ui.py
```

<a name="chinese"></a>

# 已归档的可选缩放实验（3.0.4）

**当前3.0.5已取消此功能。** 下文仅保留历史3.0.4的方法与测量，不是当前使用说明。现行UI、CLI和API均不提供缩放；历史复现需使用[v3.0.4源码](https://github.com/Montos-Qu/ecc-protected-wouaf/tree/v3.0.4)，当前规则见[图片尺寸说明](IMAGE_SIZE_LIMITS.md)。

**默认仍为原始512模式，不训练、不改权重、不改门槛。** 网页选择“图片尺寸处理→等比缩放至512 + 补边·实验”，才能接收其他尺寸；处理后预览显示实际送入模型的512x512图片。原上传文件不变。以上CLI中 `--resize fit` 为显式开启，默认不缩放。

使用Pillow LANCZOS等比缩放，必要时居中补RGB128灰边，不裁剪、不拉伸。原图已为512x512时，保留原始编码字节，不重新转换。V3先按上传文件头选原分支，再处理像素；缩放的JPEG420仍使用V2。中间使用无损PNG，不在功能内部再做JPEG压缩。JSON记录原始/目标/内容尺寸、补边、算法、原图及处理后hash、实验标记；没有增加EXIF自动旋转。结果中的文件格式/采样描述原上传文件，不是中间PNG。

上传仍限10MiB；自动模式允许总像素最多16,777,216、任一边不超过8192。原始模式继续拒绝非512图片。历史严格参考模式仅原始尺寸，缩放通过普通portable环境运行。

## 真实标签样本回归

本机现有48个编码文件来自16张源图，包含PNG/JPEG444/JPEG420、两种码的正样本、生成负样本和自然负样本。每码每处理方式：4张正源图×3格式=12个相关正视图，8张负源图×3格式=24个负视图。这些是历史实验兼容性样本，**不是新独立队列，也不是432张独立图片**。

六种场景×两种码×36视图，共432次真实推理。72个原图512的自动模式结果，去掉新增预处理元数据后，与原始模式完整结果完全一致，原始编码字节保留。五项HTTP测试以及浏览器的原始模式拒绝、可选缩放、等比补边、处理后预览、语言切换和结果下载通过。

| 处理方式 | k11正确恢复 /12正样本视图 | k16正确恢复 /12正样本视图 |
|---|---:|---:|
| 原始512 | 8/12 | 5/12 |
| 原图512，自动模式不改像素 | 8/12 | 5/12 |
| 一次重编码对照 | 8/12 | 5/12 |
| 512→1024→自动512 | 4/12 | 2/12 |
| 512→256→自动512 | 0/12 | 0/12 |
| 加边为768x512→等比缩放/补边 | 0/12 | 0/12 |


这432条记录未观察到错误接受身份或负样本被接受；余下正视图均被拒绝。样本小且视图相关，不能据此宣称零风险或总体恢复率。原大规模科研结果不变，不能用此小样本数字替换。

重编码对照保持尺寸/容器，JPEG使用Q90及原采样。其他变体保留PNG/JPEG容器，JPEG在上传前重新编码Q90，不能忽略重编码与缩放的交互。宽图试验是在原正方形源图外加灰边后再处理，不代表验证了任意非正方形照片。协议、GT、logits、概率、指纹、拒绝原因、Hamming误码及hash均另存于上述reports链接。

**结论：功能接收尺寸的范围扩大，但实测识别变差；缩小后恢复及补边样本全部拒绝。** 本版通过的是功能和原始模式保留检查，不是几何鲁棒性达标。自动缩放保持可选实验功能，优先使用原始512图片。本轮没有根据结果训练或调门槛。复现用以上命令并选择新输出目录，旧结果不会被覆盖。

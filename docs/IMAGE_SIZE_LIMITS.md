[English](#english) | [中文](#chinese)

<a name="english"></a>

# Why automatic resizing is unavailable

The current product accepts native512x512 PNG/JPEG files and does not resize, crop or add borders. Version3.0.5 restores the3.0.3 inference behavior and removes the optional3.0.4 preprocessing feature from the UI, CLI and API. The trained V3/V1/V2 weights, codebooks, thresholds and original scientific results remain unchanged. No training was added.

Resizing a file is technically possible. The limitation is reliable fingerprint recovery: interpolation recomputes pixels, downsampling discards detail, and scaling/padding changes the spatial pattern presented to the extractor. The observed reduction below does not identify one unique physical cause or establish that every possible resizing method fails. It shows that the tested preprocessing did not preserve recovery adequately. A512x512 file created by manual conversion is accepted by the size check, but this does not make it equivalent to the original native image.

The optional-fitting regression used48 existing encoded files from16 sources, six scenarios and two codes, totaling432 inference records. For each code/scenario,12 positive views come from four source images in PNG/JPEG444/JPEG420;24 negative views come from eight negative sources. These correlated fixtures are not a fresh independent cohort or432 independent images.

| Image treatment | k11 correct positive views | k16 correct positive views |
|---|---:|---:|
| Original512 | 8/12 | 5/12 |
| Already512, optional mode bypass | 8/12 | 5/12 |
| One reencoding control | 8/12 | 5/12 |
| Enlarge to1024, fit back to512 | 4/12 | 2/12 |
| Downsample to256, fit back to512 | 0/12 | 0/12 |
| Add margins, fit and pad | 0/12 | 0/12 |

All72 complete auto512/native comparisons matched. No false accepted identity or accepted negative was observed, which does not imply zero risk. Counts are small and descriptive, not replacement estimates for the original large-cohort paper results. JPEG derivatives were reencoded at Q90 with original subsampling; the control separates an additional encoding, though interactions remain possible. The wide case added margins to an existing square image; it did not test arbitrary non-square photographs.

Use the original native512 image when available. The app cannot infer the file's earlier resizing history or reconstruct lost fingerprint information. Supporting other sizes reliably would require separate evidence and might require a different method or robustness training; this release does neither. Rejection is not proof of watermark absence.

The historical protocol/results remain under [reports/resize_v3_0_4](../reports/resize_v3_0_4/) and the [archived experiment explanation](RESIZE_EXPERIMENT_ARCHIVE.md). Reproduction of that removed feature requires the immutable [v3.0.4 source](https://github.com/Montos-Qu/ecc-protected-wouaf/tree/v3.0.4). Do not run its old fitting commands against this current version.

<a name="chinese"></a>

# 为什么不提供自动缩放

现行成品只接收原生512x512 PNG/JPEG，不缩放、不裁剪、不补边。3.0.5恢复3.0.3的推理行为，移除3.0.4的可选预处理界面、CLI和API。V3/V1/V2原权重、码本、门槛及原科学结果不变，本轮没有训练。

文件缩放在技术上可以实现，限制在于能否可靠恢复指纹：插值会重新计算像素，缩小会丢失细节，缩放/补边改变提取器看到的空间模式。下列结果没有确定唯一物理原因，也不能证明所有缩放算法都会失败；它表明已测方案没有充分保住恢复效果。手工变为512的文件可以通过尺寸检查，但不等同于原生图片。

缩放回归使用48个历史编码文件、16源图，六场景、两种码，共432条推理记录。每码每场景12个正视图来自4张源图的PNG/JPEG444/JPEG420；24个负视图来自8张负源图。它们相关且样本小，不是新独立队列或432张独立图片。上表依次显示：k11原图8/12、放大后还原4/12、缩小后还原0/12、加边后处理0/12；k16为5/12、2/12、0/12、0/12。原512不处理及单次重编码对照保持基线计数。

72组自动512/原始模式完整结果相同；未观察到错误接受身份或负样本被接受，不代表零风险。这些描述性计数不能替换原大规模实验率。JPEG变体使用Q90及原采样重编码，额外编码对照不能消除所有与缩放的交互；宽图实验仅对原正方形图加边，不代表任意非正方形照片。

有原图时请使用原生512文件。程序无法推断之前的改尺寸历史，也无法找回已经丢失的指纹信息。可靠支持其他尺寸需要单独证据，可能涉及其他方法或鲁棒性训练；本版没有进行。拒绝也不能证明图片无水印。

历史协议/结果保留在上述reports目录和[实验归档说明](RESIZE_EXPERIMENT_ARCHIVE.md)。复现已撤销功能需使用[v3.0.4源码](https://github.com/Montos-Qu/ecc-protected-wouaf/tree/v3.0.4)，不要在现行版本运行旧缩放命令。

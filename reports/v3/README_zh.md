# Final V3: file-format-aware trained dual-extractor system

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

This is a system-level improvement: PNG/JPEG444 use V1, actual JPEG420 uses the trained V2 expert. Routing reads header sampling factors rather than filename, GT, user ID or test labels. BCH, confidence and presence policies remain fixed.

The extra1600-step single-model attempt and subsequent weight interpolation failed simultaneous retention and remain separate controls. This routed combination adds no training updates and does not claim one network improved every format.

Historical4096-source regression cohort,1024 positives per code/condition: this study reencoded views, checked actual headers and pixels, and executed12288 network forwards.

| Code | Condition | V1 correct | V3 correct | Change |
|---|---|---:|---:|---:|
| BCH(31,11)+P | PNG |930/1024 (90.82%)|930/1024 (90.82%)|0.00 pp|
| BCH(31,11)+P | JPEG444 Q90 |897/1024 (87.60%)|897/1024 (87.60%)|0.00 pp|
| BCH(31,11)+P | JPEG420 Q90 |755/1024 (73.73%)|787/1024 (76.86%)|+3.12 pp|
| BCH(31,16)+P | PNG |729/1024 (71.19%)|729/1024 (71.19%)|0.00 pp|
| BCH(31,16)+P | JPEG444 Q90 |597/1024 (58.30%)|597/1024 (58.30%)|0.00 pp|
| BCH(31,16)+P | JPEG420 Q90 |333/1024 (32.52%)|393/1024 (38.38%)|+5.86 pp|

PNG/444 per-image bits and acceptance decisions exactly match V1. False attributions including wrong positive identities and unwatermarked acceptance, summed across conditions: k11 V1 0 -> V3 0; k16 V1 1 -> V3 1. The combined fixed selection/calibration/regression check passed.

The router needs the actual encoded file. A JPEG decoded/saved as PNG no longer identifies its former420 sampling and uses V1. An RGB tensor alone cannot supply reliable sampling provenance. JPEG422/grayscale/other formats conservatively use V1 without the same effectiveness claim. Quantification covers512x512 Pillow Q90 444/420; other qualities/sizes require separate tests.

The development routing rule followed observed V1/V2 and failed single-model results. Selection/calibration and the4096-source cohort had been seen historically and are not a fresh independent test. PNG/444 retention follows the same V1 and pixels, not a single-model generalization result. Two fixed checkpoints are stored, one network runs per image. See the current root README for later fresh independent validation and default product use.

### Loading trained models

Current product entry, from the repository root:

```bash
python wouaf.py setup
python wouaf.py predict --image YOUR_ACTUAL_FILE.jpg --code 11 --output outputs/v3.json
```

The historical raw-delivery workflow extracted `V3_FORMAT_ROUTED_DELIVERY.zip` into `experiments/v3_format_router` in the earlier V1 portable repository, then used its `format_router_v3.py` through the locked launcher. That archived path is a historical reference, not required by the current product.

The router reads actual headers without a `--condition` argument, extra JPEG encoding or silent resizing. Output records version/branch, actual container/sampling, weights SHA256, logits/probabilities/bits and attribution. k16 is the65536-user control.

The old V1 `infer.py --condition jpeg_q90_420/444` actively reencoded at Q90; its default `png8_control` read actual files. The current V3 entry only reads encoded files. A description of that argument in the failed single-model raw delivery was incorrect; this explanation supersedes it.

The raw delivery retained both trained weights, inference code, per-image records, protocols, audits, demo and restore notebook. No retraining is needed. Full1600-step optimizer/RNG/checkpoints and failed candidates remain in the separate historical directory. Large original runtime/generator assets were not duplicated into that raw experiment; the current product obtains its required public runtime automatically. Original historical V1/V2 assets remain unchanged.

---

<a name="chinese"></a>

## 中文

历史实验记录；当前成品默认版本见仓库根README中的最终V3。以下保留该阶段的方法和结果。

## V3：按文件格式自适应的已训练双提取器

本版为系统层优化：PNG及JPEG444使用V1，实际JPEG420文件使用已训练V2专家。路由读取文件头中的采样因子，不看文件名、GT、用户ID或测试标签。BCH、置信度与presence门槛完全固定。

V3追加1600步的单模型训练与后续权重融合未同时满足保留要求，均单独保留。当前组合没有新增训练更新，也不声称某一个模型同时提升了全部格式。

以下为旧4096源图回归队列，每种条件、每个编码有1024张正样本；本轮重新编码、检查实际文件头、核对全部像素并重新执行了12288个神经网络前向。

| 编码 | 条件 | V1正确恢复 | V3正确恢复 | 变化 |
|---|---|---:|---:|---:|
| BCH(31,11)+P | PNG | 930/1024 (90.82%) | 930/1024 (90.82%) | +0.00个百分点 |
| BCH(31,11)+P | JPEG444 Q90 | 897/1024 (87.60%) | 897/1024 (87.60%) | +0.00个百分点 |
| BCH(31,11)+P | JPEG420 Q90 | 755/1024 (73.73%) | 787/1024 (76.86%) | +3.12个百分点 |
| BCH(31,16)+P | PNG | 729/1024 (71.19%) | 729/1024 (71.19%) | +0.00个百分点 |
| BCH(31,16)+P | JPEG444 Q90 | 597/1024 (58.30%) | 597/1024 (58.30%) | +0.00个百分点 |
| BCH(31,16)+P | JPEG420 Q90 | 333/1024 (32.52%) | 393/1024 (38.38%) | +5.86个百分点 |

PNG/444逐样本预测bit与接收/拒绝决策均与V1完全一致。错误归属（正样本认错用户及无水印图误接收，三条件合计）：
- k11: V1 0 → V3 0。
- k16: V1 1 → V3 1。

固定selection/calibration及回归联合检查：通过。

限制：该方案需要原始编码文件。JPEG先被解码并另存为PNG后，PNG文件头无法揭示过去的420压缩，将走V1；只有RGB张量也没有可靠采样元数据，不能承诺相同收益。JPEG422、灰度JPEG及其他格式保守走V1，未作效果保证。当前量化结果对应512×512、Pillow Q90的444/420条件，其他质量或尺寸需独立验证。

实验解释：路由规则是在观察V1/V2及V3单模型失败结果后提出的开发方案。selection/calibration和4096源图均已在历史实验观察，不能称为新独立测试。PNG/444保留来自使用相同V1模型与像素，不是单模型训练的泛化证明。需要两个固定checkpoint，单张图只执行其中一个网络。

### 加载已训练模型

将V3_FORMAT_ROUTED_DELIVERY.zip解压到原V1便携仓库新目录experiments/v3_format_router。按V1 README恢复相同冻结环境后，在仓库根目录运行：

```bash
python scripts/run_locked.py experiments/v3_format_router/format_router_v3.py --image YOUR_ACTUAL_FILE.jpg --code 11 --output outputs/v3.json
```

本入口从实际文件头自动选分支，不需要--condition，也不会额外JPEG重编码或静默缩放。输出同时记录model_version、selected_branch、实际container/sampling、权重SHA、logits、概率、bits和归属决策。k=16对应65,536用户对照。

注意原V1的infer.py：其--condition jpeg_q90_420/444参数会主动额外Q90重编码；直接读取真实文件应使用其默认png8_control。本V3自动入口只读取文件，不重新压缩。旧单模型V3交付包LOAD_V3_DELIVERY.md对该参数的描述有误，以本说明为准。

V3_FORMAT_ROUTED_DELIVERY.zip包含两份已训练权重、自动推理代码、逐样本GT/logits/概率/bit/决策、协议、审计、演示和恢复notebook。无需再次训练。完整1600步训练的检查点/optimizer/RNG及失败候选仍保留在上级V3目录；不复制15GB原环境/生成器资产，环境从原V1 release/offline_assets恢复。V1/V2和原默认配置保持不变。

# ECC-Protected WOUAF - Pretrained V3 Product

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

**Download the product, run the setup command, and use the trained final V3. No retraining is required.** The tool reads native 512x512 PNG/JPEG files, extracts WOUAF user fingerprints, and returns an accepted user attribution or `REJECT` when evidence is insufficient.

Code, configuration and selection for V1, V2 and V3 are included. V3 is the default final product; V1/V2 are optional controls. Training source is separate in `training/` and is never invoked by normal setup or inference.

### 1. Download and use

**Local web interface (recommended):** double-click `start_ui.cmd`. On first use it runs automatic setup if needed, then opens WOUAF Studio in your default browser. Upload/drop a native 512x512 PNG/JPEG, choose V3 (default), V1 or V2, and select **Identify fingerprint**. The page shows the accepted user ID or a plain-language rejection reason, the actual extractor/format, and a downloadable JSON. English/Chinese switching and built-in samples are included. Use **Stop app** to stop the local server; closing the tab alone leaves it running. Images are processed on this computer; no cloud service is used. See [local UI guide](docs/LOCAL_UI.md).

```bash
python wouaf.py ui                  # Open the installed portable product in a browser.
python wouaf.py ui --no-browser     # Print the loopback URL without opening a browser.
```

**Windows without Python:** extract the complete product ZIP and run `setup_windows.cmd`. It bootstraps a project-local Python automatically. Then drag a 512x512 image onto `predict_windows.cmd`, or use `predict_windows.cmd "YOUR_IMAGE.png" v3` in a terminal. The optional second argument selects `v1` or `v2`. Prediction JSON is saved under `outputs/`. No system Python or A100 is needed.

Download the **complete product ZIP** from the [latest Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/latest), or clone this repository / use GitHub Code -> Download ZIP. Enter the extracted project directory:

```bash
python wouaf.py setup
python wouaf.py predict --image YOUR_IMAGE.png --output outputs/prediction.json
```

`setup` automatically downloads/verifies trained checkpoints and support files, creates an isolated managed Python environment, installs fully locked and hash-verified official PyTorch packages, and checks the trained model. It does not change system Python. The complete product ZIP already contains weights and support files. No retraining or login to the author's Drive is required.

**Ordinary local use is now the default. An A100 is not required.** Supported product platforms are Windows 10/11 x64 and x86_64 Linux (glibc2.28+). Any working host Python3.9+ can start setup; it automatically manages Python3.12.12, torch2.7.1, torchvision0.22.1, NumPy1.26.4, Pillow12.3.0 and SciPy1.15.3. Users do not locate/install packages one by one.

```bash
python wouaf.py setup                              # Select NVIDIA CUDA when usable, otherwise CPU.
python wouaf.py predict --image YOUR_IMAGE.png
python wouaf.py setup --device cpu                 # Explicit local CPU installation.
python wouaf.py setup --device cuda                # Require a working supported NVIDIA GPU/driver.
python wouaf.py predict --image YOUR_IMAGE.png --device cpu
```

Automatic GPU setup selects official CUDA11.8 or CUDA12.8 PyTorch wheels for the detected NVIDIA generation. RTX50/compute capability10+ selects CUDA12.8. `--cuda-backend cu118` or `--cuda-backend cu128` can override selection. A supported NVIDIA driver is required for GPU acceleration; setup does not replace drivers. In automatic mode, an unusable CUDA runtime selects CPU. CPU mode also works on PCs with AMD graphics; this product does not claim AMD GPU acceleration. Hardware is checked by an actual tensor operation and a model self-check, rather than an A100 model-name restriction.

**A100 is retained only for historical strict reference reproduction**, through an explicit option:

```bash
python wouaf.py setup --runtime reference
python wouaf.py predict --runtime reference --image YOUR_IMAGE.png
```

Reference mode preserves the original Linux Python3.10.21 / torch1.12.1+cu116 archive and A100-SXM4-80GB / driver580.82.07 checks. Portable mode uses the same trained weights, preprocessing, BCH codebooks, routing and rejection thresholds, with a separately pinned runtime; it does not label its output as strict A100 reference execution. See [portable runtime details](docs/PORTABLE_RUNTIME.md) and [reference environment](docs/ENVIRONMENT.md). macOS/ARM inference is outside the current executable profiles.


The [public product notebook](01_PUBLIC_V3_REPRODUCE.ipynb) runs the same portable setup without mounting the author's Drive; A100 selection is optional. `00_PRETRAINED_V3_DEMO.ipynb` remains the historical author-Drive restore workflow.

Try a supplied input and run the self-check:

```bash
python wouaf.py predict --image demo/images/png8_control/positive_k11/10000000.png
python wouaf.py verify
```

### 2. Choose V1, V2 or V3

| Version | Behavior and trade-off | Role |
|---|---|---|
| **V3, default** | Actual JPEG420 -> trained V2 expert; PNG/JPEG444 -> V1; one forward per image | **Final product; recommended default** |
| V1 | Trained balanced extractor for every format | Optional balanced baseline/control |
| V2 | Trained JPEG420 expert for every format; better 420 with PNG/444 trade-offs | Optional expert/control |

```bash
python wouaf.py versions
python wouaf.py predict --image YOUR_IMAGE.jpg
python wouaf.py predict --image YOUR_IMAGE.jpg --version v1
python wouaf.py predict --image YOUR_IMAGE.jpg --version v2
```

The default BCH(31,11)+overall parity supports 2048 users. Add `--code 16` for the 65536-user capacity control. All versions use fixed BCH, parity and rejection policies; inference never trains or calibrates them. Routing uses the **actual encoded file header**, not filenames, GT or test labels.

### 3. Trained weights and training requirements

Weights are learned network parameters stored in `.pth` files. Code defines the architecture; the weights supply the behavior learned during training. **Actual trained weights are provided and downloaded automatically, so consumers do not retrain the model.** Missing/corrupt files fail verification instead of silently using a randomly initialized network.

V3 combines two trained components; it does not require a third separately trained V3 `.pth`:

| File | Used by | Original SHA256 |
|---|---|---|
| `models/v1_negative_aware_2400.pth` | V1; V3 PNG/JPEG444 branch | `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb` |
| `models/v2_jpeg420_expert_extra800.pth` | V2; V3 JPEG420 branch | `7a5f5b5b3149dd4ebd2837e3a36367fbb3cda4691935ca017c24fd403b48827d` |

### 4. Inputs and outputs

`ACCEPT` means the candidate passed correction and rejection rules; `user_id` is the accepted identity. `REJECT` returns `user_id: null`; an internal candidate must not be treated as a valid attribution. JSON also records version/branch, model hash, 32-bit fingerprint, logits/probabilities and rejection reason.

The validated scope is native 512x512 RGB, PNG, and Pillow JPEG Q90 4:4:4 / 4:2:0. No silent resizing occurs. Actually decoding a compressed JPEG and saving it as PNG removes historical JPEG sampling metadata, so V3 uses V1 for that new PNG. Other sizes, compression conditions and attacks require separate validation.

### 5. Public environment and data

These seven research asset groups/nine files are read-only link-public. **Ordinary portable setup does not download the archived A100 environment.** Full independent-image evaluation adds the six image groups; the archived environment is used only with `--runtime reference`. The private parent release folder is not needed.

| Public asset group | Download | Compressed size |
|---|---|---:|
| Frozen runtime: three parts | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| Native JPEG420 views | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |
| Native JPEG444 views | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| Generated negative PNGs | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| Natural negative PNGs | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| Positive k11 PNGs | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| Positive k16 PNGs | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |

Requested research-data downloads default to `offline_assets/`; select another destination with `--asset-root PATH`. Environment parts total 2,319,751,708 bytes; environment plus the independent images totals 4,829,967,179 bytes. Extraction needs additional space. Exact file URLs, sizes and SHA256 values are in [public_assets.json](public_assets.json).

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
```

To recompute statistics from saved per-image predictions without new network forwards, see the [full public reproduction guide](docs/PUBLIC_REPRODUCTION.md). Original training data, the full generator and historical 5000 latents remain optional research assets; **loading the trained product and rerunning this fixed independent image cohort does not require them**.

### 6. Final-version results

Fresh independent-content validation, 1024 positives per code/condition, V1 -> final V3:

| Code | PNG accepted exact recovery | JPEG444 Q90 | JPEG420 Q90 |
|---|---:|---:|---:|
| k11, default | 90.14% -> 90.14% | 87.60% -> 87.60% | **74.12% -> 76.86%** |
| k16, capacity control | 67.68% -> 67.68% | 56.35% -> 56.35% | **33.40% -> 37.40%** |

The JPEG420 gain has a paired scene-cluster 95% interval of +1.37 to +4.10 percentage points for k11, and +2.64 to +5.47 for k16. PNG/444 outputs remain identical per sample. Zero false attribution was observed in this finite study, not a guarantee of zero population risk. Full new-cohort V2 results across all formats are not presented as part of the original final validation table; save additional evaluations separately. Complete V1/V2/V3 controls, training history and limitations are in [RESULTS](docs/RESULTS.md) and the [research workflow](docs/RESEARCH_REPRODUCTION_zh.md).

### 7. Project layout and optional training

```text
wouaf.py                         Product setup / predict / versions / verify / evaluate
configs/pretrained.json          Default final V3 system
configs/versions/v1.json,v2.json Optional V1/V2 controls
models/                          Checkpoint destinations and verified download manifest
ecc_wouaf/                       Extractor, format routing, BCH, parity, rejection
environment/locked/              Frozen runtime restorer, inventories and isolated launcher
public_assets.json               Public runtime/image file URLs, sizes and hashes
training/                        Separate training source, unused by normal inference
reports/                         Controls, training history and per-image results
docs/                            Runtime, workflow, training and reproduction details
01_PUBLIC_V3_REPRODUCE.ipynb      Public product setup without private Drive mounting
```

Only consumers who want to change/retrain models need the separate [training workflow](docs/TRAINING.md). `setup` automatically restores saved predictions, native demo files and binary decoding tables from support assets. Large objects use GitHub Releases/public Drive rather than ordinary Git history.

Preserve upstream [WOUAF](https://github.com/kylemin/WOUAF) licensing and [third-party notices](THIRD_PARTY_NOTICES.md). Public downloads do not replace original model, image and library terms. Product-distribution updates do not retrain models, modify weights or tune thresholds. Windows CPU migration was tested on 48 native fixtures: zero bit differences and zero changed accepted attribution decisions versus saved A100 outputs; maximum logit difference was 0.0000362396240234375. This finite fixture check does not prove identical behavior on the entire cohort or on every GPU. NVIDIA CUDA and Linux installation paths use official binaries but were not hardware-tested in this Windows/AMD session.

---

<a name="chinese"></a>

## 中文

**下载、运行环境配置命令，然后直接使用。默认是已经训练好的最终 V3，不需要重新训练。** 本工具读取 512×512 的原生 PNG/JPEG 图片，恢复 WOUAF 用户指纹，并输出用户归属或 `REJECT`。证据不足时拒绝归属。

V1、V2、V3 的代码、配置和版本选择均保留；V1/V2 是额外选项。训练源码单独位于 `training/`，普通安装和预测不会进入训练。

### 1. 下载并使用

**本地网页界面（推荐）：**双击 `start_ui.cmd`。首次需要时会自动配置环境，然后在默认浏览器打开 WOUAF Studio。上传/拖入原生512x512 PNG/JPEG，选择默认V3或V1/V2，点击“开始识别指纹”。直接显示接受的用户ID或直白的拒绝原因、实际提取器/格式，并可下载JSON。支持中英文切换及内置示例。点击“关闭服务”停止本机进程；仅关闭浏览器标签页不会停止服务。图片全部在本机处理，不使用云服务。见[界面说明](docs/LOCAL_UI.md)。

```bash
python wouaf.py ui                  # Open the installed portable product in a browser.
python wouaf.py ui --no-browser     # Print the loopback URL without opening a browser.
```

**Windows没有预装Python也可以：**解压完整产品ZIP后运行 `setup_windows.cmd`，它会自动准备项目独立的Python环境。然后把512×512图片拖到 `predict_windows.cmd` 上，或在终端执行 `predict_windows.cmd "YOUR_IMAGE.png" v3`；第二个参数也可选择v1或v2。结果JSON保存在 `outputs/`。不需要系统Python，也不需要A100。

从 [最新 Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/latest) 下载 **完整产品 ZIP** 并解压；也可 clone 本仓库或使用 GitHub 的 Code → Download ZIP。进入项目目录后：

```bash
# Download trained checkpoints, demo assets and the frozen runtime; restore and self-check.
python wouaf.py setup

# Use final V3 on a native 512x512 PNG/JPEG file.
python wouaf.py predict --image YOUR_IMAGE.png --output outputs/prediction.json
```

`setup` 自动下载/校验训练权重与支撑文件，创建独立的 Python 环境，按完整版本与hash锁文件安装官方 PyTorch 依赖，然后检查模型。不修改系统 Python。完整产品 ZIP 已包含权重和支撑文件，安装时直接复用。**不需要逐个对照清单安装依赖，不需要重新训练，也不用登录作者的 Drive。**

**普通个人电脑本地使用已设为默认，不要求 A100。** 支持 Windows10/11 x64、x86_64 Linux（glibc2.28+）。用已有的 Python3.9+ 启动安装器即可，它自动管理 Python3.12.12、torch2.7.1、torchvision0.22.1、NumPy1.26.4、Pillow12.3.0 和 SciPy1.15.3。

```bash
python wouaf.py setup                              # Prefer usable NVIDIA CUDA; otherwise select CPU.
python wouaf.py predict --image YOUR_IMAGE.png
python wouaf.py setup --device cpu                 # Explicit CPU installation.
python wouaf.py setup --device cuda                # Require a supported NVIDIA GPU and driver.
python wouaf.py predict --image YOUR_IMAGE.png --device cpu
```

自动配置按显卡代际选择官方 CUDA11.8 或 CUDA12.8 PyTorch 安装包，RTX50/compute capability10+ 使用 CUDA12.8；也可添加 `--cuda-backend cu118` 或 `--cuda-backend cu128` 明确选择。GPU 加速需要可用的 NVIDIA 驱动，程序不替换系统驱动。自动模式下 CUDA 无法工作时改用 CPU。配有 AMD 显卡的电脑也可运行 CPU 模式，但本版不声称支持 AMD GPU 加速。检查依据是真实张量运算与模型自检，不再把显卡名称必须为 A100 当作普通使用条件。

**A100 只作为原实验的严格参考环境保留**，需要主动选择：

```bash
python wouaf.py setup --runtime reference
python wouaf.py predict --runtime reference --image YOUR_IMAGE.png
```

严格参考模式继续保留原 Linux Python3.10.21 / torch1.12.1+cu116 归档以及 A100-SXM4-80GB / driver580.82.07 检查。普通模式使用同一份训练权重、预处理、BCH 码表、格式路由和拒绝门槛，通过单独锁定的软件栈在本地运行；输出不会冒充 A100 严格复现。详见[普通运行模式](docs/PORTABLE_RUNTIME.md)和[参考环境说明](docs/ENVIRONMENT.md)。当前没有提供 macOS/ARM 的可执行推理环境。


不想复制终端命令时，使用 [公开产品 notebook](01_PUBLIC_V3_REPRODUCE.ipynb)：它直接从公开 GitHub/Drive 文件恢复，不挂载作者 Drive。原 `00_PRETRAINED_V3_DEMO.ipynb` 保留为作者原 Drive 归档的历史恢复入口。

先用随附图片验证输入：

```bash
python wouaf.py predict --image demo/images/png8_control/positive_k11/10000000.png
python wouaf.py verify
```

### 2. V1、V2、V3 怎么选

| 版本 | 行为与区别 | 用途 |
|---|---|---|
| **V3，默认** | 真实 JPEG420 → V2 专家；PNG/JPEG444 → V1；每图一次网络前向 | **最终产品，推荐日常使用** |
| V1 | 所有格式都用已训练的平衡提取器 | 额外的平衡基线/对照 |
| V2 | 所有格式都用已训练 JPEG420 专家；420 改善，PNG/444 有取舍 | 额外的专家/对照 |

```bash
python wouaf.py versions
python wouaf.py predict --image YOUR_IMAGE.jpg                 # Final V3 by default.
python wouaf.py predict --image YOUR_IMAGE.jpg --version v1
python wouaf.py predict --image YOUR_IMAGE.jpg --version v2
```

默认 BCH(31,11)+整体 parity，容量 2048 用户；需要容量对照时添加 `--code 16`，容量 65536 用户。所有版本使用固定 BCH、parity 和拒绝门槛，预测时不会校准或训练。V3 按**实际文件头**识别 JPEG420，不按文件名或 GT 选分支。

### 3. 模型权重是什么，需不需要训练

权重是训练得到的网络参数，保存在 `.pth` 文件中；代码定义网络，权重提供训练学到的行为。**本项目提供真实训练权重，安装器自动取得它们，所以使用者无需重新训练。** 缺失、损坏或校验不符会报错，不会把随机初始化网络当作成品。

V3 是两份已训练组件组成的系统，不是第三份单独训练的 `.pth`：

| 文件 | 使用版本 | 原始 SHA256 |
|---|---|---|
| `models/v1_negative_aware_2400.pth` | V1，以及 V3 的 PNG/JPEG444 分支 | `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb` |
| `models/v2_jpeg420_expert_extra800.pth` | V2，以及 V3 的 JPEG420 分支 | `7a5f5b5b3149dd4ebd2837e3a36367fbb3cda4691935ca017c24fd403b48827d` |

权重与实验支撑附件在 [v3.0.0 模型 Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/tag/v3.0.0) 保留不变；当前产品安装器使用这些已校验的原始对象。产品安装/下载流程更新不代表追加训练了一个新模型版本。

### 4. 读取结果

`ACCEPT` 表示通过纠错与拒绝规则，`user_id` 是接受的用户 ID。`REJECT` 时 `user_id` 为 `null`，内部候选不能当作有效归属。JSON 还保留所选版本/分支、模型 hash、32-bit 指纹、logits/probabilities 和拒绝原因。

验证范围为原生 512×512 RGB、PNG 与 Pillow JPEG Q90 4:4:4 / 4:2:0。输入不会被静默缩放。把压缩 JPEG 真正转存成 PNG 会丢失原 JPEG 采样历史，V3 此时按新的 PNG 文件头使用 V1；其他尺寸、压缩条件和攻击需要另做验证。

### 5. 公开环境和数据下载

下面 7 组、9 个研究资产允许匿名只读下载。**普通本地安装不下载旧 A100 环境归档**；完整独立图像验证才额外下载六组图片。旧环境归档只在 `--runtime reference` 模式中使用，不需要访问作者的私有发布根目录。

| 公开文件组 | 下载链接 | 下载大小 |
|---|---|---:|
| 冻结 Python、PyTorch 与全部依赖（3 个分片） | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| k11 水印 PNG 正样本 | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| k16 水印 PNG 正样本 | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |
| 同源无水印 PNG 负样本 | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| 自然图 PNG 负样本 | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| 真实 JPEG444 Q90 验证视图 | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| 真实 JPEG420 Q90 验证视图 | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |

可选研究数据的默认下载位置为 `offline_assets/`，也可通过 `--asset-root PATH` 指定。演示环境归档共 2,319,751,708 字节；完整环境加独立图像共 4,829,967,179 字节，解压需额外空间。每个分片的 URL、大小和 SHA256 在 [public_assets.json](public_assets.json)。

```bash
# Prepare the product and complete independent images without training.
python wouaf.py setup --with-evaluation

# Evaluate V3 on native images and save new outputs.
python wouaf.py evaluate --version v3 --out outputs/independent_v3
# Optional controls on the same image cohort.
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
```

无需重新前向时，可按 [完整复现指南](docs/PUBLIC_REPRODUCTION.md) 从公开保存的逐样本预测重算统计。旧训练数据、完整生成器、历史 5000 latents 等属于可选研究资产，仍独立保存；**加载成品和重跑上述独立图像验证不需要这些资产**。

### 6. 最终版本的结果

以下为新独立内容验证，每码、每条件 1024 张正样本，V1 → 最终 V3：

| 码 | PNG 正确恢复 | JPEG444 Q90 | JPEG420 Q90 |
|---|---:|---:|---:|
| k11，默认 | 90.14% → 90.14% | 87.60% → 87.60% | **74.12% → 76.86%** |
| k16，容量对照 | 67.68% → 67.68% | 56.35% → 56.35% | **33.40% → 37.40%** |

JPEG420 改善的 95% 场景块配对区间：k11 +1.37～4.10 个百分点，k16 +2.64～5.47 个百分点。PNG/444 逐样本输出保持一致。本次有限样本中观察到错误归属为 0，不代表总体风险为 0。V2 在所有新格式的全量结果不冒充原最终验证表；自行评估应另存。详细的 V1/V2/V3 对照、训练过程与局限见 [RESULTS](docs/RESULTS.md) 和 [完整研究流程](docs/RESEARCH_REPRODUCTION_zh.md)。

### 7. 项目结构与可选自行训练

```text
wouaf.py                         成品入口：setup / predict / versions / verify / evaluate
configs/pretrained.json          默认最终 V3
configs/versions/v1.json,v2.json  可选 V1/V2
models/                          真实权重的目标路径及自动下载清单
ecc_wouaf/                       提取、格式路由、BCH、parity 与拒绝逻辑
environment/locked/              冻结环境恢复器、清单和隔离启动器
public_assets.json               已公开环境/验证图像的实际下载与校验信息
training/                        单独的训练源码；普通使用不会调用
reports/                         版本对照、训练历史与逐样本实验结果
docs/                            完整环境、研究流程、训练和复现说明
01_PUBLIC_V3_REPRODUCE.ipynb      不挂载私人 Drive 的成品入口
```

自行训练仅供需要改变模型的使用者，说明在 [TRAINING](docs/TRAINING.md)。保存预测、演示图与二进制表由 `setup` 自动从支撑附件恢复。大文件采用 GitHub Releases / 公开 Drive 分发，不放进普通 Git 历史。

保留上游 [WOUAF](https://github.com/kylemin/WOUAF) 许可与 [第三方通知](THIRD_PARTY_NOTICES.md)。公开下载不改变原模型、照片和归档库的既有许可。本次本地兼容改造没有重训、修改权重或调整门槛。Windows CPU 已实测48张原生测试图片：与保存的A100结果相比，指纹位差异为0、接受的归属决定变化为0，最大logit差异0.0000362396240234375。这是有限测试，不能据此声称全部队列或所有显卡逐位一致。本机是Windows/AMD配置，本次未对NVIDIA CUDA或Linux硬件进行实测；其安装入口使用官方二进制。

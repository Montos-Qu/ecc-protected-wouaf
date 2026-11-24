# ECC-Protected WOUAF — Pretrained V3

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

**A ready-to-use tool for reading WOUAF fingerprints from images. Download it, set it up, and use the trained V3 model. You do not need to train anything.**

If an image contains a recoverable WOUAF fingerprint, the app can report its user ID. If it cannot recover enough reliable information, it shows `REJECT` and explains why. This is an image-reading tool: it does not add fingerprints to your photos, detect every kind of watermark, or identify the author of any ordinary picture.

### 1. Start here: Windows web app

1. Open the [latest Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/latest) and download the **complete product ZIP** whose name ends in `local-3.0.6-final.zip`. Choose this file rather than GitHub's automatically generated source ZIP if you want weights and demo files included.
2. Extract the whole ZIP into a normal folder. Do not run files from inside the ZIP.
3. Double-click `start_ui.cmd`. The first run downloads Python and installs the required packages in this project folder. Keep the window open until setup finishes. Internet is needed for this first setup; you do not need to preinstall Python, use Colab, log into the author's Drive, or have an A100.
4. Your default browser opens WOUAF Studio. Upload a **native512x512 PNG or JPEG**, keep the default **V3**, and click **Identify fingerprint**.
5. Read the user ID or rejection reason. **Download result JSON** saves the full result. A copy is also kept in `outputs/ui/`.

To check that your installation works, upload `demo/images/png8_control/positive_k11/10000000.png`. With the default V3/k11 settings, the supplied positive example should return **user565**. An ordinary photo may be rejected; that alone is not an installation problem. A watermarked JPEG can also be rejected if its fingerprint is too damaged.

Use **Stop app** to stop the local service. Closing the browser tab alone leaves it running. Starting the same project again reopens its existing service; a different project copy uses another free local port if necessary. Images are processed on your computer, without a cloud upload. The page supports English and Chinese and includes a larger desktop layout. See the [UI guide](docs/LOCAL_UI.md).

### 2. What pictures work, and what do the results mean?

- Use original512x512 PNG/JPEG files, no larger than10MiB. The app does **not** resize, crop, or add borders. Resizing can damage the fingerprint even if the final file is512x512. See [why automatic resizing was removed](docs/IMAGE_SIZE_LIMITS.md).
- `ACCEPT`: the recovered fingerprint passed the model's fixed checks. `user_id` is the accepted ID.
- `REJECT`: the checks did not pass, so no user is attributed and `user_id` is `null`. This does not prove the image has no watermark. The page explains the first failed check, such as too many bit errors or insufficient evidence.
- The JSON includes the chosen version, the extractor actually used, image format, model-file checksum,32 predicted bits, raw model scores (`logits`), bit probabilities, and the rejection reason. Bit probabilities are not the probability that an identity is correct.

The measured accuracy applies to original512x512 RGB images, PNG and Pillow JPEG Q90 with4:4:4/4:2:0 sampling. Other qualities, editing, screenshots and attacks are not covered by those numbers. Converting a JPEG to PNG does not undo compression and also removes its original JPEG sampling information.

### 3. V3, V1 and V2: which should I choose?

| Choice | What it does | When to use it |
|---|---|---|
| **V3, default** | Uses V1 for PNG/JPEG444 and V2 for actual JPEG420 files | **Recommended final product** |
| V1 | Uses the balanced extractor for every image | Compare with the earlier balanced version |
| V2 | Uses the JPEG420 specialist for every image | Compare JPEG420 results; PNG/444 results may be worse |

V3 chooses by the actual file header, not its extension or a known answer. It stores two trained components and runs one per image. It does not need a separate third V3 weight file.

The default error-correcting code, BCH(31,11)+overall parity (`k11`), supports2048 user IDs. The optional `k16` setting supports65536 IDs but recovers fewer images in our tests. These are capacity choices, not different amounts of training. Start with V3 and k11.

### 4. Prefer commands?

Windows without Python can run `setup_windows.cmd` and then drag a picture onto `predict_windows.cmd`, or run:

```bat
predict_windows.cmd "YOUR_IMAGE.png" v3
predict_windows.cmd "YOUR_IMAGE.jpg" v1
predict_windows.cmd "YOUR_IMAGE.jpg" v2
```

If you already have Python3.9+, clone/download the repository, open a terminal in its folder, and run:

```bash
python wouaf.py setup
python wouaf.py ui
python wouaf.py predict --image YOUR_IMAGE.png --output outputs/prediction.json
python wouaf.py predict --image YOUR_IMAGE.jpg --version v1
python wouaf.py predict --image YOUR_IMAGE.jpg --version v2
python wouaf.py predict --image YOUR_IMAGE.png --code 16
python wouaf.py versions
python wouaf.py verify
```

`setup` installs the exact locked package versions automatically; no manual comparison with a package list is needed. It also downloads missing trained weights and support files, checks their hashes, and tests the model. The complete product ZIP already includes those project files; the source ZIP/clone obtains them during setup. Setup does not train a model or change your system Python. After setup, inference works locally. The [public notebook](01_PUBLIC_V3_REPRODUCE.ipynb) is an optional alternative without mounting the author's Drive.

### 5. Computer requirements and environment

Supported installation profiles: Windows10/11 x64 and x86_64 Linux with glibc2.28+. An NVIDIA GPU can accelerate inference; CPU mode also works on a computer with AMD graphics. AMD GPU acceleration, macOS and ARM are not included in these profiles.

```bash
python wouaf.py setup --device cpu
python wouaf.py setup --device cuda
python wouaf.py predict --image YOUR_IMAGE.png --device cpu
python wouaf.py ui --no-browser
```

The project manages Python3.12.12, PyTorch2.7.1, torchvision0.22.1, NumPy1.26.4, Pillow12.3.0 and SciPy1.15.3. All dependencies are pinned and checked against hashes. Automatic setup tries usable NVIDIA CUDA, otherwise CPU. It selects official CUDA11.8 or12.8 packages for the GPU generation; RTX50/compute capability10+ selects12.8. `--cuda-backend cu118` or `--cuda-backend cu128` overrides that choice. A compatible NVIDIA driver is required; setup does not install drivers. See [runtime details](docs/PORTABLE_RUNTIME.md).

The components were trained on an A100. **Ordinary users do not need an A100 or the original training environment.** Local mode uses the same weights and decision rules with its own fixed package versions. Windows CPU was tested against saved A100 outputs on48 native fixtures: no changed fingerprint bits or accepted IDs; maximum raw-score difference0.0000362396240234375. This finite check does not prove identical results for every image or GPU. NVIDIA CUDA and Linux paths were not hardware-tested in this Windows/AMD session.

For researchers who need the historical A100 environment, use `python wouaf.py setup --runtime reference` and `python wouaf.py predict --runtime reference --image YOUR_IMAGE.png`. That separate mode checks Linux Python3.10.21, torch1.12.1+cu116, A100-SXM4-80GB and driver580.82.07. See [reference environment](docs/ENVIRONMENT.md). Normal setup does not download that archive.

### 6. Results and optional research downloads

On a new independent image set, with1024 watermarked images per code and format, the share correctly recovered and accepted was:

| Capacity | PNG: V1 → V3 | JPEG444 Q90: V1 → V3 | JPEG420 Q90: V1 → V3 |
|---|---:|---:|---:|
| k11, default |90.14% →90.14%|87.60% →87.60%|**74.12% →76.86%**|
| k16, larger capacity |67.68% →67.68%|56.35% →56.35%|**33.40% →37.40%**|

V3 improves JPEG420 while keeping PNG/444 outputs identical to V1 for this test. The paired scene-cluster95% interval for the420 gain is +1.37 to +4.10 percentage points for k11 and +2.64 to +5.47 for k16. No false user attribution was observed in this finite study; that is not a guarantee of zero risk. The original final table does not claim a full new-set V2 evaluation across every format. Earlier V1/V2 comparisons, failed attempts and limits are kept in [results](docs/RESULTS.md) and the [research workflow](docs/RESEARCH_REPRODUCTION_zh.md).

**You do not need the following large research downloads for normal use.** They are public, read-only downloads for replaying experiments. The old environment has3 parts; the six image groups are for full independent evaluation:

| Public asset group | Download | Compressed size |
|---|---|---:|
| Frozen runtime: three parts | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| Native JPEG420 views | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |
| Native JPEG444 views | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| Generated negative PNGs | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| Natural negative PNGs | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| Positive k11 PNGs | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| Positive k16 PNGs | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |

Exact file URLs, sizes and SHA256 checksums are in [public_assets.json](public_assets.json). The environment parts total2,319,751,708 bytes; environment plus images total4,829,967,179 bytes, with extra disk space needed for extraction. Optional downloads go to `offline_assets/`, or `--asset-root PATH`.

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
```

See [full reproduction steps](docs/PUBLIC_REPRODUCTION.md) to use saved per-image predictions or rerun image inference. The original generator, training data and5000 latents are separate research assets; the trained app and this fixed-image evaluation do not need them. `00_PRETRAINED_V3_DEMO.ipynb` is the historical author-Drive restore notebook, not the ordinary public setup.

### 7. Files, trained weights and optional training

| Folder/file | Purpose |
|---|---|
| `start_ui.cmd`, `wouaf.py` | Normal app and command-line entry points |
| `models/` | Already-trained V1/V2 component weights; V3 uses both |
| `configs/`, `ecc_wouaf/`, `ecc/` | Version choices, image reading, bit recovery and fixed acceptance rules |
| `environment/portable/` | Local environment installer and dependency locks |
| `environment/locked/` | Historical reference environment support |
| `training/` | Optional code for people who want to train a different model |
| `reports/`, `docs/` | Experiments, results, limits and reproduction guides |
| `outputs/` | Your local results and installation records |

**Weights** are the parameters learned during training, saved as`.pth` files. The provided weights make this a trained product. If they are missing, setup downloads them; it does not ask users to retrain or silently use a random model. The original files stay available in the [v3.0.0 model Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/tag/v3.0.0):

| File | SHA256 checksum |
|---|---|
| `models/v1_negative_aware_2400.pth` | `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb` |
| `models/v2_jpeg420_expert_extra800.pth` | `7a5f5b5b3149dd4ebd2837e3a36367fbb3cda4691935ca017c24fd403b48827d` |

Training is separate and optional; see [training instructions](docs/TRAINING.md). Normal setup, the UI and prediction commands never start it. Large files use GitHub Releases/public Drive instead of ordinary Git history.

### 8. Original WOUAF and credits

This Applied Project builds on [kylemin's original WOUAF repository](https://github.com/kylemin/WOUAF), accompanying the CVPR2024 work. Our additions cover error correction, parity and rejection tests, V1/V2/V3 comparisons, local setup and the web UI. Credit for the original method and implementation remains with its authors. Use the [upstream citation](https://github.com/kylemin/WOUAF#citation) for academic references; see [references](docs/REFERENCES.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Public downloads do not replace the original model, image or library license terms.

---

<a name="chinese"></a>

## 中文

**这是一个下载后就能使用的 WOUAF 图片指纹识别工具。默认加载训练好的 V3，无须重新训练。**

如果图片里保留了能恢复的 WOUAF 指纹，程序会给出对应用户ID；信息不足时显示 `REJECT`，并说明原因。它负责读取已有指纹，不会给照片添加指纹，也不是通用水印检测器，不能识别任意普通照片的作者。

### 1. Windows 用户从这里开始

1. 打开[最新 Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/latest)，下载名称以 `local-3.0.6-final.zip` 结尾的**完整产品 ZIP**。它已经包含训练权重和演示文件；GitHub 自动生成的源码 ZIP 不包含这些大文件。
2. 把整个 ZIP 解压到普通文件夹，不要在压缩包内直接运行。
3. 双击 `start_ui.cmd`。首次会自动下载 Python 并安装依赖，所有环境文件都放在项目目录里。请等配置结束。首次需要联网；不用预装 Python、不用 Colab、不用登录作者的 Drive，也不用 A100。
4. 默认浏览器会打开 WOUAF Studio。上传**原始512×512的 PNG/JPEG**，保留默认 **V3**，点击**开始识别指纹**。
5. 查看用户ID或拒绝原因。点击**下载完整 JSON 结果**可保存详细结果，程序也会在 `outputs/ui/` 留一份副本。

想先确认安装正常，可以上传 `demo/images/png8_control/positive_k11/10000000.png`。保持默认V3/k11，随附的这张带指纹示例应显示**用户565**。普通照片被拒绝不代表安装失败；带指纹JPEG压缩后，也可能因为指纹受损而被拒绝。

点击**关闭服务**才会停止后台；只关浏览器标签页不会停止。再次启动同一目录会打开已有服务，另一个项目副本遇到端口占用则自动换空闲端口。图片只在本机处理，不上传云端。界面支持中英文和较大的桌面字号，详见[网页界面说明](docs/LOCAL_UI.md)。

### 2. 能上传什么图片？结果怎么看？

- 使用原始512×512 PNG/JPEG，上限10MiB。程序**不缩放、不裁剪、不补边**。手动改成512也不能恢复缩放时丢失的指纹信息，见[为什么取消自动缩放](docs/IMAGE_SIZE_LIMITS.md)。
- `ACCEPT`：恢复的指纹通过了固定检查，`user_id` 是被接受的用户ID。
- `REJECT`：没有通过检查，因此不归属给任何用户，`user_id` 为 `null`。这不能证明图片没有水印。页面会解释最先失败的规则，例如错误位太多，或证据不足。
- JSON保留所选版本、实际使用的提取器、图片格式、模型文件校验值、32位预测指纹、原始模型分数（`logits`）、每位概率和拒绝原因。每位概率不等于“用户身份正确的概率”。

实验准确率对应原始512×512 RGB、PNG，以及Pillow Q90的JPEG444/JPEG420。其他压缩质量、编辑、截图或攻击不包含在这些准确率里。把JPEG转存为PNG既不能消除压缩损伤，也会丢失原JPEG采样信息。

### 3. V3、V1、V2 怎么选？

| 选项 | 做什么 | 适合用途 |
|---|---|---|
| **V3，默认** | PNG/JPEG444用V1，真实JPEG420用V2 | **最终成品，日常推荐** |
| V1 | 所有图片都用平衡提取器 | 与早期平衡版本对比 |
| V2 | 所有图片都用JPEG420专家 | 对比JPEG420效果；PNG/444可能下降 |

V3读取真实文件头来选择模型，不看扩展名或已知答案。项目保存两份训练组件，每张图片只运行其中一份；不需要第三个单独的V3权重文件。

默认纠错方式BCH(31,11)+整体奇偶校验（`k11`）支持2048个用户ID。可选 `k16` 支持65536个ID，但本实验中的成功恢复率较低。这是用户容量的取舍，不是训练步数的区别。一般保留V3和k11即可。

### 4. 想用命令行？

没有Python的Windows用户可先运行 `setup_windows.cmd`，再把图片拖到 `predict_windows.cmd` 上，也可执行：

```bat
predict_windows.cmd "YOUR_IMAGE.png" v3
predict_windows.cmd "YOUR_IMAGE.jpg" v1
predict_windows.cmd "YOUR_IMAGE.jpg" v2
```

已经安装Python3.9+的用户，可以克隆或下载仓库，在项目目录打开终端执行：

```bash
python wouaf.py setup
python wouaf.py ui
python wouaf.py predict --image YOUR_IMAGE.png --output outputs/prediction.json
python wouaf.py predict --image YOUR_IMAGE.jpg --version v1
python wouaf.py predict --image YOUR_IMAGE.jpg --version v2
python wouaf.py predict --image YOUR_IMAGE.png --code 16
python wouaf.py versions
python wouaf.py verify
```

`setup` 自动按锁定版本安装依赖，不用对照清单逐个下载。它还会补齐训练权重和支撑文件，校验文件是否完整，并测试模型。完整产品ZIP已包含这些项目文件；源码ZIP/克隆版会在安装时补齐。安装不会训练模型，也不修改系统Python。配置完成后在本机推理；[公开 notebook](01_PUBLIC_V3_REPRODUCE.ipynb)只是另一种可选入口，不需要挂载作者Drive。

### 5. 电脑要求与环境配置

目前提供Windows10/11 x64和x86_64 Linux（glibc2.28+）安装方案。有兼容NVIDIA显卡可以加速；配AMD显卡的电脑也能使用CPU模式。本版不包含AMD显卡加速、macOS或ARM运行方案。

```bash
python wouaf.py setup --device cpu
python wouaf.py setup --device cuda
python wouaf.py predict --image YOUR_IMAGE.png --device cpu
python wouaf.py ui --no-browser
```

程序自动管理Python3.12.12、PyTorch2.7.1、torchvision0.22.1、NumPy1.26.4、Pillow12.3.0、SciPy1.15.3；所有依赖都有固定版本和文件校验。自动配置优先使用可工作的NVIDIA CUDA，否则改用CPU。按显卡代际选择官方CUDA11.8/12.8包，RTX50/compute capability10+使用12.8；也可用 `--cuda-backend cu118` 或 `--cuda-backend cu128` 指定。显卡驱动需要用户已有兼容版本，程序不安装驱动。详见[本地运行环境](docs/PORTABLE_RUNTIME.md)。

组件是在A100上训练的，**普通使用不需要A100，也不需要重建训练环境**。本地模式保留原权重和判断规则，使用单独固定的软件版本。Windows CPU已与保存的A100结果对比48张原始图片：指纹位和已接受用户ID均无变化，最大原始分数差0.0000362396240234375。有限测试不能证明所有图片、所有显卡都完全一致。本次Windows/AMD测试没有实测NVIDIA CUDA或Linux硬件。

只有需要复现历史A100实验的研究者，才使用 `python wouaf.py setup --runtime reference` 和 `python wouaf.py predict --runtime reference --image YOUR_IMAGE.png`。该模式检查Linux Python3.10.21、torch1.12.1+cu116、A100-SXM4-80GB和driver580.82.07，见[参考环境说明](docs/ENVIRONMENT.md)。普通安装不会下载这份旧环境归档。

### 6. 实验结果与可选研究数据

在一组新的独立图片上，每种编码、每种格式各1024张带指纹图片，成功恢复正确用户并通过检查的比例为：

| 容量设置 | PNG：V1 → V3 | JPEG444 Q90：V1 → V3 | JPEG420 Q90：V1 → V3 |
|---|---:|---:|---:|
| k11，默认 |90.14% →90.14%|87.60% →87.60%|**74.12% →76.86%**|
| k16，较大容量 |67.68% →67.68%|56.35% →56.35%|**33.40% →37.40%**|

这次测试里V3提高了JPEG420效果，PNG/444的逐图输出与V1完全一致。按同源场景分组计算，420提升的95%配对区间为：k11 +1.37～+4.10个百分点，k16 +2.64～+5.47个百分点。本次有限测试未观察到错误用户归属，不代表真实风险为零。原最终表没有声称完成V2在新数据全部格式上的全量测试。早期V1/V2对照、失败尝试和限制保留在[结果说明](docs/RESULTS.md)及[研究流程](docs/RESEARCH_REPRODUCTION_zh.md)。

**日常使用不用下载下面这些大型研究数据。** 它们开放匿名只读下载，供复现实验；旧环境有3个分片，6组图片用于完整独立评估：

| 公开文件组 | 下载链接 | 下载大小 |
|---|---|---:|
| 冻结 Python、PyTorch 与全部依赖（3 个分片） | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| k11 水印 PNG 正样本 | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| k16 水印 PNG 正样本 | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |
| 同源无水印 PNG 负样本 | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| 自然图 PNG 负样本 | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| 真实 JPEG444 Q90 验证视图 | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| 真实 JPEG420 Q90 验证视图 | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |

准确下载地址、大小和SHA256校验值见 [public_assets.json](public_assets.json)。环境分片合计2,319,751,708字节；环境加图片合计4,829,967,179字节，解压还需要额外空间。可选文件默认下载到 `offline_assets/`，也可用 `--asset-root PATH` 指定位置。

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
```

[完整复现指南](docs/PUBLIC_REPRODUCTION.md)介绍如何用已保存的逐图预测重算统计，以及如何重新读取图片评估。原生成器、训练数据和5000个latent是单独的研究资产，使用训练好的成品和这组固定图片评估不需要它们。`00_PRETRAINED_V3_DEMO.ipynb` 是历史作者Drive恢复入口，不是普通用户安装步骤。

### 7. 项目文件、训练权重与自行训练

| 文件/目录 | 用途 |
|---|---|
| `start_ui.cmd`、`wouaf.py` | 日常网页和命令行入口 |
| `models/` | 训练好的V1/V2组件权重，V3使用这两份组件 |
| `configs/`、`ecc_wouaf/`、`ecc/` | 版本选择、读取图片、恢复指纹和固定判断规则 |
| `environment/portable/` | 本地环境安装器和依赖锁定文件 |
| `environment/locked/` | 历史参考环境相关文件 |
| `training/` | 想训练其他模型时使用的可选代码 |
| `reports/`、`docs/` | 实验、结果、限制和复现说明 |
| `outputs/` | 本机结果与安装记录 |

**权重**就是训练学到的参数，保存在`.pth`文件里。项目提供的权重让它成为训练好的产品；缺失时安装器自动下载，不会让用户重新训练，也不会悄悄使用随机模型。原文件继续保存在 [v3.0.0模型Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/tag/v3.0.0)：

| 文件 | SHA256校验值 |
|---|---|
| `models/v1_negative_aware_2400.pth` | `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb` |
| `models/v2_jpeg420_expert_extra800.pth` | `7a5f5b5b3149dd4ebd2837e3a36367fbb3cda4691935ca017c24fd403b48827d` |

自行训练是独立的可选功能，见[训练说明](docs/TRAINING.md)。普通配置、网页和预测命令都不会启动训练。大文件通过GitHub Releases或公开Drive分发，不放入普通Git历史。

### 8. 原始 WOUAF 与引用

本Applied Project基于[kylemin的WOUAF原项目](https://github.com/kylemin/WOUAF)，对应CVPR2024工作。我们的扩展包括纠错、奇偶校验与拒绝测试、V1/V2/V3对照、本地安装和网页界面。原方法与实现归原作者。学术引用请使用[上游Citation](https://github.com/kylemin/WOUAF#citation)，并阅读[引用说明](docs/REFERENCES.md)和[第三方通知](THIRD_PARTY_NOTICES.md)。开放下载不改变原模型、图片和依赖库的许可条件。

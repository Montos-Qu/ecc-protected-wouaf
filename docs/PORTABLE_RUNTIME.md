[English](#english) | [中文](#chinese)

<a name="english"></a>

# Portable local inference

Use `python wouaf.py setup`, then `python wouaf.py predict --image YOUR_IMAGE.png`. Windows x64 and Linux x86_64 use pinned official PyTorch binaries; A100 is not required. V3 remains default, V1/V2 use `--version`, and both original checkpoints are reused without training.

The bootstrap downloads a SHA256-verified uv0.9.5 executable from PyPI, manages Python3.12.12 within the project runtime directory, and installs explicit SHA256-pinned official torch2.7.1/torchvision0.22.1 wheels plus pinned common dependencies. No administrator-level Python installation or global package upgrade occurs. The resolved complete inventory is saved locally and checked on each launch. Compiled runtime directories are excluded from Git and release archives.

| Mode | Selection | Requirements |
|---|---|---|
| Automatic, default | `setup` | Uses supported NVIDIA CUDA when detected; otherwise CPU |
| CPU | `setup --device cpu` | No NVIDIA GPU needed; works on a PC with AMD graphics |
| NVIDIA | `setup --device cuda` | Working NVIDIA driver and hardware supported by the selected official wheel |
| Exact historical reference | `setup --runtime reference` | Original Linux archive, A100-SXM4-80GB and driver580.82.07 |

CUDA11.8 is selected for earlier NVIDIA generations; CUDA12.8 for RTX50/compute capability10+. Use `--cuda-backend cu118` or `--cuda-backend cu128` for an explicit backend. Setup verifies CUDA by a real tensor/conv operation; automatic mode falls back to CPU if that probe fails. A forced CUDA request reports failure rather than claiming GPU execution. Old/unsupported NVIDIA hardware may only work through CPU. This does not provide AMD GPU acceleration, macOS/ARM inference, or a guarantee for every NVIDIA model/driver combination.

After installation, `predict --device cpu` can run in CPU mode even with a CUDA-enabled environment. To add GPU libraries after a CPU-only install, rerun `setup --device cuda`. No retraining occurs. `verify` runs native fixtures and compares routing, weights, bit outputs and accepted identities with saved references. Setup only reports model readiness after this self-check passes; it records numerical differences rather than claiming bitwise floating-point equivalence.

On this Windows/AMD machine, CPU inference passed48 native fixtures and48 fixed V1/V2 branch controls: zero changed fingerprint bits, zero changed accepted identities, maximum saved-A100 logit difference3.62396240234375e-05. GPU/Linux paths are implemented but were not hardware-tested here. Full scientific tables remain original A100-reference results, not newly measured portable-cohort metrics.

Training/generation continue to use their documented reference recipes. The portable product targets trained extractor inference and existing-image evaluation. It does not silently migrate historical training protocols or change decoder thresholds.

Official sources: [PyTorch binary versions](https://pytorch.org/get-started/previous-versions/), [uv managed Python](https://docs.astral.sh/uv/guides/install-python/). See `environment/portable/profile.json` for pinned artifact URLs/hashes and `outputs/portable_install.json` for the machine's local installation receipt.


<a name="chinese"></a>

# 个人电脑本地推理

运行 `python wouaf.py setup`，再运行 `python wouaf.py predict --image YOUR_IMAGE.png`。Windows x64 和 Linux x86_64 使用锁定版本的官方 PyTorch 安装包，不要求 A100。默认最终V3，V1/V2用 `--version` 选择，两份原训练权重直接复用，不训练。

启动器从PyPI取得经过SHA256校验的uv0.9.5，在项目运行目录中管理Python3.12.12，并安装已固定SHA256的官方torch2.7.1/torchvision0.22.1及锁定的公共依赖。不需要管理员级安装Python，也不升级系统全局包。完整实际安装清单保存于本机，每次启动都核验。运行时二进制目录不会打进Git或产品ZIP。

默认自动选择可用的NVIDIA CUDA，否则CPU。`setup --device cpu`不要求NVIDIA卡，AMD显卡电脑也可使用CPU；`setup --device cuda`要求可用的NVIDIA驱动及所选官方包支持的显卡。`setup --runtime reference`才使用原Linux/A100-SXM4-80GB/driver580.82.07严格环境。

较早显卡自动选CUDA11.8，RTX50/compute capability10+选CUDA12.8；可用 `--cuda-backend cu118/cu128`明确选择。安装时执行真实张量/卷积运算检查，自动模式在CUDA检查失败时回退CPU；显式请求CUDA则报告失败。旧/不支持的NVIDIA卡可能只能使用CPU。本版不提供AMD GPU加速或macOS/ARM推理，也不承诺所有NVIDIA显卡/驱动组合都支持加速。

安装后，CUDA环境也能用 `predict --device cpu`在CPU运行。CPU安装后想启用GPU，再执行 `setup --device cuda`即可，不需要训练。`verify`通过真实文件检查路由、权重、指纹和归属决定；模型自检通过后才标记为就绪，数值差异会保留记录，不冒称浮点逐位严格一致。

本机Windows/AMD的CPU实测：48张原生图片及48个V1/V2分支对照通过，指纹位变化0、接受身份变化0，最大A100参考logit差异3.62396240234375e-05。本次未实测GPU/Linux硬件。论文完整结果表仍来自原A100实验，不是本轮重新测得的完整跨平台队列指标。

训练和生成仍使用各自的参考配方；普通模式面向已训练提取器推理和现有图片评估，不擅自迁移历史训练协议，也不修改门槛。官方来源：[PyTorch](https://pytorch.org/get-started/previous-versions/)、[uv Python管理](https://docs.astral.sh/uv/guides/install-python/)。固定URL/hash见 `environment/portable/profile.json`，本机安装记录见 `outputs/portable_install.json`。

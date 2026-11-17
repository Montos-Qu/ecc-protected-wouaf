[English](#english) | [中文](#chinese)

<a name="english"></a>

# Environment configuration

## Ordinary local product, default

Run `setup_windows.cmd` on Windows, or `python wouaf.py setup` with host Python3.9+ on Windows x64/Linux x86_64. Setup creates an isolated project-local Python3.12.12 environment and installs torch2.7.1, torchvision0.22.1, NumPy1.26.4, Pillow12.3.0 and SciPy1.15.3 automatically. Six OS/backend lockfiles include complete pinned/hash-verified dependencies. System Python and NVIDIA drivers are not replaced. V3 trained weights are loaded directly; no retraining occurs.

CPU is supported without an NVIDIA GPU. A usable NVIDIA device selects official CUDA11.8/12.8 binaries, with an actual tensor and model check. The portable profile is deliberately separate from the historical reference stack. Its local receipt records software and numerical validation and marks `reference_runtime_verified: false`. See [portable setup](PORTABLE_RUNTIME.md), [public downloads](PUBLIC_REPRODUCTION.md), and `environment/portable/profile.json`.

Windows CPU was actually tested:48 native fixtures and48 V1/V2 branch controls, zero bit differences/changed accepted identities against saved A100 references, maximum logit difference3.62396240234375e-05. Windows no-system-Python setup and Unicode-path prediction also passed. NVIDIA/Linux hardware was not tested in this session. These checks do not prove entire-cohort numerical equivalence on every device.

## Historical strict reference, optional

```bash
python wouaf.py setup --runtime reference
python wouaf.py verify --runtime reference
```

This mode uses the public three-part archived Linux environment, not modern portable packages. It requires a Linux bootstrap Python3.12+, restores the archived interpreter Python3.10.21 to `/content/strict_wouaf_restore/env`, and checks91 distributions,23 conda builds, GPU model and driver. The reference is A100-SXM4-80GB /580.82.07. Package/import isolation and strict checks remain unchanged. Do not install or upgrade packages inside this frozen environment. Windows/macOS can download/check its artifacts with `setup --runtime reference --download-only`, but cannot execute its Linux binaries natively.

Machine-readable authority: `environment/locked/locked_environment_profile.json`; readable inventory: `requirements.reference.txt`. The archive previously passed a clean-directory restore and224 reference-output checks; this release does not claim another fresh Colab restore. [Public archive](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt), sizes/hashes in `public_assets.json`.

## Scientific protocol and limits

Historical extraction/current training uses float32, TF32 off, cudnn benchmark off, cudnn deterministic True, threads2. The historical frozen generator uses deterministic False. Reusing the training switch during generation caused small uint8 pixel differences; restoring the generator's original setting produced12/12 pixel-identical PNG controls. Evidence and the earlier failed check remain preserved. Match the stage-specific protocol rather than assuming every deterministic flag should be enabled.

The archive preserves the validated frozen stack. Earlier installations without saved wheel/binary hashes cannot retrospectively be proven byte-identical to the very first machine. Historical missing optional packages, incomplete pip-check state and fallback behavior are not hidden by upgrades. Validated inference and the dual-head recipe do not validate every upstream training branch. Portable mode is a new pinned profile using the same weights, thresholds and inference core; it does not claim strict original-hardware equivalence.

---

<a name="chinese"></a>

# 环境配置

## 默认个人电脑成品

Windows运行 `setup_windows.cmd`；已有Python3.9+的Windows x64/Linux x86_64运行 `python wouaf.py setup`。安装器创建独立的项目本地Python3.12.12环境，自动安装torch2.7.1、torchvision0.22.1、NumPy1.26.4、Pillow12.3.0、SciPy1.15.3。六份系统/后端依赖锁定文件包含完整的固定版本及hash。不替换系统Python或显卡驱动，直接加载已训练V3，不重新训练。

没有NVIDIA显卡也支持CPU。可用的NVIDIA设备选择官方CUDA11.8/12.8安装包，并经过真实张量和模型检查。普通模式与历史参考环境独立，本机记录保留软件及数值验证，`reference_runtime_verified`为false。见[普通模式](PORTABLE_RUNTIME.md)、[公开下载](PUBLIC_REPRODUCTION.md)及 `environment/portable/profile.json`。

Windows CPU已实测48张原生图片和48个V1/V2分支对照：与保存的A100参考相比，指纹位差异0、接受身份变化0，最大logit差异3.62396240234375e-05。不依赖系统Python的Windows安装和中文路径预测也已通过。本轮未实测NVIDIA/Linux硬件，这些检查不代表所有硬件完整队列数值一致。

## 可选历史严格参考模式

```bash
python wouaf.py setup --runtime reference
python wouaf.py verify --runtime reference
```

该模式使用公开的三分片Linux环境归档。Linux宿主Python3.12+负责启动，归档中的Python3.10.21恢复到 `/content/strict_wouaf_restore/env`，检查91个distributions、23个conda builds、GPU及驱动。参考硬件A100-SXM4-80GB/580.82.07，导入隔离及严格检查原样保留。不要在冻结环境内安装或升级包。Windows/macOS可以用 `setup --runtime reference --download-only` 下载核验，但不能原生执行Linux二进制。

权威机器清单是 `environment/locked/locked_environment_profile.json`，可读清单为 `requirements.reference.txt`。该归档此前通过空目录恢复及224个参考输出检查；本次发布不冒称再次进行了全新Colab恢复。[公开归档](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt)，大小/hash见 `public_assets.json`。

## 科学流程与边界

历史提取/当前训练使用float32、TF32 off、cudnn benchmark off、cudnn deterministic True、threads2；冻结生成器的历史流程使用deterministic False。曾把训练开关带到生成阶段，造成少量uint8像素差异；恢复生成器原设置后12/12 PNG控制逐像素一致。原失败与修复证据均保留。复现应匹配每个阶段的原设置。

归档保留的是已验证冻结栈。更早未保存wheel/binary hash的安装，无法事后证明与最初机器逐字节相同。历史可选包缺失、pip check不完整和fallback状态未通过升级掩盖；已验证推理及双头训练不等于全部上游训练分支都验证。普通模式是使用相同权重、门槛、推理核心的新锁定环境，不声称原硬件严格等同。

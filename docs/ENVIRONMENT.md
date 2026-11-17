# 环境配置与严格复现的边界

## 已冻结并保留

`environment/locked/locked_environment_profile.json` 是机器可读权威清单，`requirements.reference.txt` 是可读清单。归档含已安装的解释器、Python 包和原生依赖；模型推理通过 `run_wouaf_locked.py` 的隔离解释器运行。默认 Colab Python 只负责挂载 Drive、解压和启动，不参与科学推理。

严格模式核对 Python、解释器前缀、91 个 distributions、23 个 conda build、GPU 型号、驱动，并关闭用户 site/PYTHONPATH 注入。参考 A100-SXM4-80GB / driver580.82.07。恢复目标为固定 `/content/strict_wouaf_restore/env`；仓库文件无需放在固定路径。不要在该冻结环境内安装或升级包。

```bash
python scripts/materialize_assets.py --asset-root "$ASSETS" --id frozen_environment
python scripts/restore_environment.py
python scripts/run_locked.py --check
python scripts/run_locked.py scripts/verify_pretrained.py
```

恢复脚本检查归档及文件清单；已有环境必须一致才继续，不会为了“能跑”自动忽略检查。归档先前做过空目录恢复与 224 个参考输出检查；本次打包中的迁移回归是在当前已恢复参考运行时上执行，不能冒称本轮又做了一次全新的 Colab runtime 恢复。

## 确定性配置是流程的一部分

提取/当前训练：float32，TF32 off，cudnn benchmark off，cudnn deterministic **True**，torch threads2。固定旧生成器的历史解码流程：cudnn deterministic **False**。此前误把训练的 True 带入生成阶段，引起少量 uint8 像素 ±1 差异；恢复生成器原设置后 12/12 PNG 控制逐像素一致。原失败记录和修复证据保留，不把它隐藏成“没有发生”。

不要把“确定性开关越多越好”当作复现原则；应忠实恢复该阶段原本的设置。

## 为什么不声称与项目最初机器绝对相同

这份归档可以严格保留**当前已验证的冻结软件栈**。对于早先没有保存原始 wheel/binary hash 的历史安装，无法事后证明每个二进制都与最初那台机器逐字节相同。原有缺少 matplotlib / pytorch-msssim、pip check 未全通过、缺 ninja 的参考 fallback 等状态没有通过升级掩盖；已运行的推理和本次双头训练配方经过实际验证，但不等于所有上游训练分支均已验证。

CPU、不同 GPU、不同驱动或现代 PyTorch 的迁移应建立新 profile 和独立回归记录。`--allow-unvalidated-runtime` 只为明确选择非参考模式提供入口，不会把结果标成 reference verified。

# GitHub发布结构

- 默认推理系统仍为已训练V3，V1/V2由 `--version` 显式选择。训练源码独立保留在 `training/`。
- 普通Git仓库保存代码、配置、详细README、环境概要和结果表；原生演示图、逐样本实验预测、ECC二进制表和较大的环境/数据清单作为带SHA256的支撑数据附件，通过 `python scripts/download_support.py` 恢复到原路径。完整项目ZIP也包含全部这些文件。
- 大权重采用GitHub Releases，不使用LFS占位符。运行 `python scripts/download_pretrained.py`，按 `models/pretrained_manifest.json` 下载并校验两份真实权重。
- 同一Release另提供包含实际权重的完整预训练项目ZIP、SHA256校验文件与历史Drive归档来源记录。
- 完整冻结环境、baseline生成器、原始图像和latents等约16.7GiB的大资产继续存放在原Drive目录。README给出入口和 `asset_manifest.json` 部件校验。它们沿用原有访问权限；尚未取得访问权限的用户需向项目所有者申请，不能仅凭clone声称已恢复严格环境。
- 旧未清理notebook、原研究备份和历史V1/V2/V3归档保持原样。随仓库演示notebook无执行输出，基于Drive原归档恢复。
- 原始Drive主ZIP的SHA256是 `6480004217d32758d0eb5e496689fbda2e15798c2e775b3accea6fb45de74ca7`。本次只调整下载分发和发布文档，没有训练或修改模型、规则、推理数值逻辑与实验结果。
- 上游LICENSE和第三方通知保留；公开可见不代表把所有第三方资产统一改成某个开源许可。

下载模型后运行 `python scripts/verify_files.py` 验证项目文件；严格环境恢复后按README执行演示和对照。硬件/驱动不匹配仍需显式使用未验证模式，不会默默绕过检查。

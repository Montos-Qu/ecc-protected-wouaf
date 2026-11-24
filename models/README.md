# Trained model files

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

These two`.pth` files contain what the models learned during training. They are already trained: users do not need to train them again.

From the project root, run `python wouaf.py setup`. It checks existing files and downloads missing ones automatically. The complete product ZIP already includes both weights; GitHub's source ZIP/clone downloads them during setup. Large weights are stored in GitHub Releases rather than ordinary Git history.

| Component | How it is used |
|---|---|
| V1, trained2400 steps | Balanced image reader; used by V1 and by V3 for PNG/JPEG444 |
| V2, selected at800 extra steps | JPEG420 specialist; used by V2 and by V3 for JPEG420 |

V3 chooses between these components from the actual image format. There is no missing third V3 file. Both use the original architecture and fixed decision thresholds.

For a weights-only download, run `python scripts/download_pretrained.py` from the root. `pretrained_manifest.json` records the official project Release URLs, sizes and SHA256 checksums. A matching file is reused. A conflicting file causes an error without being overwritten; the app never substitutes a random untrained model.

---

<a name="chinese"></a>

## 中文

这两份`.pth`文件保存了模型训练学到的参数，也就是“权重”。它们已经训练好，使用者不用重新训练。

在项目根目录运行 `python wouaf.py setup` 即可自动检查已有文件并补齐缺失权重。完整产品ZIP已经包含两份权重；源码ZIP/克隆版会在配置时自动下载。大权重保存在GitHub Releases，不放进普通Git历史。

| 组件 | 怎么使用 |
|---|---|
| V1，训练2400步 | 平衡提取器；V1使用，V3处理PNG/JPEG444时也使用 |
| V2，选用追加800步的权重 | JPEG420专家；V2使用，V3处理JPEG420时也使用 |

V3按真实图片格式选择组件，不是缺少了第三个V3文件。组件保留原模型结构和固定判断门槛。

如果只想下载权重，在根目录执行 `python scripts/download_pretrained.py`。`pretrained_manifest.json`保存项目Release下载地址、大小和SHA256校验值。正确文件直接复用，不符文件报错但不覆盖；程序不会拿随机未训练模型代替。

# Trained model components

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

V3 uses two already trained checkpoint files. No retraining is required. Ordinary Git history excludes these large objects; their actual bytes are available in the GitHub Release.

```bash
python scripts/download_pretrained.py
```

Normal consumers can run `python wouaf.py setup` from the repository root; it includes this download step. `pretrained_manifest.json` records the actual Release URLs, file sizes and original SHA256 values. Correct existing files are reused. Conflicting existing files are rejected without overwrite, and inference never falls back to random weights. The complete pretrained Release ZIP already contains both checkpoints, so the downloader verifies those files.

V1 is the negative-aware dual-head extractor trained for 2400 steps. V2 is the JPEG420 expert selected at extra800. V3 selects the trained component from the actual file header; it is not a newly trained single network. Names, architectures, thresholds and hashes remain consistent with the original Drive archive.

---

<a name="chinese"></a>

## 中文

V3默认使用这两份已经训练好的权重，不需要重新训练。普通Git历史不存储大权重；GitHub Release提供实际文件。

```bash
python scripts/download_pretrained.py
```

下载器使用 `pretrained_manifest.json` 的真实Release地址、文件大小和原始SHA256。已有正确文件直接复用；已有不符文件会报错，不会覆盖，也不会回退到随机模型。完整预训练Release ZIP已经包含权重，下载器只做校验。

V1：负样本感知双头提取器，2400步。V2：JPEG420专家，选择extra800。V3根据实际文件头选择分支；这不是一个新训练的单模型。两版本的文件名、网络、门槛和hash均与原Drive归档一致。

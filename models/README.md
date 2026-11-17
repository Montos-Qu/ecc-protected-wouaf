# 已训练模型 / Pretrained components

V3默认使用这两份已经训练好的权重，不需要重新训练。普通Git历史不存储大权重；GitHub Release提供实际文件。

```bash
python scripts/download_pretrained.py
```

下载器使用 `pretrained_manifest.json` 的真实Release地址、文件大小和原始SHA256。已有正确文件直接复用；已有不符文件会报错，不会覆盖，也不会回退到随机模型。完整预训练Release ZIP已经包含权重，下载器只做校验。

V1：负样本感知双头提取器，2400步。V2：JPEG420专家，选择extra800。V3根据实际文件头选择分支；这不是一个新训练的单模型。两版本的文件名、网络、门槛和hash均与原Drive归档一致。

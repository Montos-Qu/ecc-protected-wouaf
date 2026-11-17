# 公开复现与下载指南

默认入口是已经训练好的 V3 产品。推荐从仓库根目录执行 `python wouaf.py setup`，然后 `python wouaf.py predict --image YOUR_IMAGE.png`。安装器自动完成以下命令，不需要使用者逐个安装依赖。

## 下载级别

| 目标 | 自动入口 | 额外资产 |
|---|---|---|
| 使用 V3 / 选择 V1/V2 / 小量自检 | `python wouaf.py setup` | 公开权重、支撑附件、冻结环境3分片 |
| 重跑完整新独立图像队列 | `python wouaf.py setup --with-evaluation` | 再加六组原生图片，共4.83GB环境/图像资产 |
| 从保存预测重算统计 | `analysis/recompute.py`，在所选运行环境中执行 | 已由setup恢复的支撑附件，无须下载完整图像 |
| 从latent再生成/自行训练 | 独立 `generation/` / `training/` 研究入口 | 额外研究资产，非成品使用必需 |

## 手动编排等价命令

下面是安装器的展开形式，供需要自定义路径的用户查看；不是要求逐个找包。Linux宿主Python3.12+只负责下载/解压，科学计算始终使用归档的Python3.10.21。

```bash
python scripts/download_pretrained.py
python scripts/download_support.py
python scripts/download_public_assets.py --preset demo
python scripts/materialize_assets.py --asset-root offline_assets --id frozen_environment
python scripts/verify_files.py
python scripts/restore_environment.py
python scripts/run_locked.py --check
python scripts/run_locked.py scripts/verify_pretrained.py --output outputs/demo_verification.json
```

`--preset images`只获取六组图片，`--preset independent`获取环境与图片全部9个文件。下载器只使用匿名HTTP会话，不读取浏览器凭据；Google的大文件下载确认由脚本处理，SHA256/文件大小不符会报错，不会把错误HTML页面当作有效文件。已有正确文件复用，已有不符文件拒绝覆盖。

## 实际公开链接

| 公开文件组 | 下载链接 | 下载大小 |
|---|---|---:|
| 冻结 Python、PyTorch 与全部依赖（3 个分片） | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| k11 水印 PNG 正样本 | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| k16 水印 PNG 正样本 | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |
| 同源无水印 PNG 负样本 | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| 自然图 PNG 负样本 | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| 真实 JPEG444 Q90 验证视图 | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| 真实 JPEG420 Q90 验证视图 | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |

| 文件 | 文件入口 | 精确字节数 | SHA256 |
|---|---|---:|---|
| `frozen_environment/frozen_environment.part000` | [下载](https://drive.google.com/file/d/1sMErk4a_C8ZK6w3i5EauY1Lz0Rqjo2iD/view) | 1073741824 | `3ff86ba424249644f44b2cbd1e26a3b35bc51f4cb03aaf46ee3407e93e6a368c` |
| `frozen_environment/frozen_environment.part001` | [下载](https://drive.google.com/file/d/1NfI0n6RExdjsnao_1X8DBYdf6Z9Zi9b_/view) | 1073741824 | `aec5f627b697ce4c9a9bec4f5276bde5b74fe5b74c69cb7695738702d4a3d232` |
| `frozen_environment/frozen_environment.part002` | [下载](https://drive.google.com/file/d/1HNy9xBzNpYTaYWj4w3qsYnD9sdXNZTKo/view) | 172268060 | `2504c964cd9a68c3df896180d460233593f859f764ba0094c824e35ff7a208b9` |
| `independent_source_pngs_positive_k11/source_pngs_positive_k11.zip` | [下载](https://drive.google.com/file/d/1z5PhK0Nzh3c9Blnshn_C7CZC9PsYJ1cw/view) | 458231358 | `5b19e774d6752202184386e7792ce0a19865ebe33b7f209be90a9958da63a7cd` |
| `independent_source_pngs_positive_k16/source_pngs_positive_k16.zip` | [下载](https://drive.google.com/file/d/17I4z-4WXCXmxXWOwOZDP8jbFAZQidwMU/view) | 458263430 | `31a1356342449bf1cae4815ec73a7d35dd32630adcedc4fedba45eea3e087e3e` |
| `independent_source_pngs_generated_negative/source_pngs_generated_negative.zip` | [下载](https://drive.google.com/file/d/14hQBofKtDUBQWSbzNwudQMlyHdNA-1qt/view) | 472933398 | `c8c82b0f41532089f909aacaa6e39eea376eebe435d87158317c45e822ae6fb4` |
| `independent_source_pngs_natural_negative/source_pngs_natural_negative.zip` | [下载](https://drive.google.com/file/d/1ybyhh1xrULcDJ6pnMyvNweFm8jRHHii9/view) | 409355808 | `bf62e115490a2b64b213b62a895660c8e253962a7a4b02f1a25abf7b8c0d3cb2` |
| `independent_encoded_jpeg_q90_444/encoded_jpeg_q90_444.zip` | [下载](https://drive.google.com/file/d/1ZlUaxakbTBsyVOtmKpP-c5nEqZTIjqSe/view) | 396389848 | `9bb9f9b535c5f6ed3e29e631739176c49646eeb52fc38a56ed525e9267fb1dac` |
| `independent_encoded_jpeg_q90_420/encoded_jpeg_q90_420.zip` | [下载](https://drive.google.com/file/d/1sNMYit3B050-lx2GZL_vHpKKYFZst09G/view) | 315041629 | `e85aa98b9322efcd5a039b602196c13a4615a16b5c7e26089c8add34fb84b01f` |

## 完整图像验证

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python scripts/run_locked.py analysis/recompute.py --v1 outputs/independent_v1 --v3 outputs/independent_v3 --out outputs/recomputed_independent
```

队列是4096源图/12288原生PNG或真实JPEG视图，输入manifest、GT、prompt和保存预测来自原独立测试记录。读取已编码的图片，不额外压缩、重采样或重新生成。V2可用相同evaluate命令作为另行保存的对照。默认主码k11，容量对照k16，门槛已冻结。

## 环境核验、版本与边界

软件环境归档、全部分片hash和恢复脚本均可下载；不能只执行 `pip install` 某个现代版本就声称严格复现。`setup`恢复归档并自动检查91个Python distributions、23个native builds，以及参考GPU/驱动。严格模式仍要求A100-SXM4-80GB / 580.82.07，环境固定前缀 `/content/strict_wouaf_restore/env`。其他平台/硬件的执行需独立迁移验证；当前分发补全没有启动Colab或重新进行GPU推理/训练。

模型和原实验对象仍是v3.0.0冻结版本；成品安装包装可有补丁版本更新。原Drive归档、v3.0.0 Release和训练权重保持不变。必要环境/验证数据通过独立子文件夹开放，作者其他备份、训练和生成器资产的访问权限没有扩大。

Google Drive可能存在下载配额；临时受限应稍后重试或通过README文件链接下载到相同 `offline_assets/<组>/<文件>` 路径，再重新运行setup。任何重新下载仍会进行相同SHA256验证。第三方许可见仓库通知；公开文件不会被自动改授统一新许可。

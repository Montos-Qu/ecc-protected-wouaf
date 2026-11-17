# 在V1便携仓库加载V3候选

尚未达到全部联合目标；保留为实验候选，原V1发布版继续保留。

将V3_DELIVERY.zip解压到新V1副本的experiments/v3。保留所有原V1文件，不修改默认configs/pretrained.json。先按V1 README恢复严格环境，再运行：

```bash
python scripts/run_locked.py infer.py --config experiments/v3/inference_config.json --image YOUR_IMAGE.png --condition jpeg_q90_420 --code 11 --output outputs/v3.json
```

`--condition`描述实际图像处理条件，不会自动压缩输入；相机/JPEG文件按其实际内容输入。JPEG验证使用Pillow质量90、4:2:0或4:4:4缓存；BCH、presence和bit门槛均固定。代码16可作65,536用户对照。

完整复核见V3_FINAL_REVIEW_zh.md。此ZIP是已训练模型；训练源码、原始plan和角色说明在experiment/，完整中间检查点在Drive原目录，手动选择运行训练脚本才会追加训练。

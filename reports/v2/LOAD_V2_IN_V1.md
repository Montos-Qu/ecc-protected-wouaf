# 在便携推理包中显式加载V2

将本候选包解压到V1便携仓库中的 experiments/v2/，不替换 models/ 下的V1权重或 configs/pretrained.json。然后在V1仓库根目录运行：

```bash
python scripts/run_locked.py infer.py --config experiments/v2/inference_config.json --image YOUR_512_IMAGE.png --code 11 --output outputs/v2_prediction.json
```

inference_config.json引用同目录的selected_multitask.pth并校验完整SHA256；BCH及环境配置使用原V1根目录的同一套冻结文件。默认不带--config仍加载V1。归属结果需同时保留所用配置和权重hash以区分版本。此包是独立研究候选，尚未自动替换V1发布版；regression_promising不等于全新独立评估通过。

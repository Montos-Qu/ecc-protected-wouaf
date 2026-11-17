# V2扩展训练：最终复核

版本：v2_codeword_jpeg420_20261029T042333Z
完成追加训练：2400步；选择的额外步数：800。

当前状态：独立实验候选，未替换V1发布模型。

| 编码 | 条件 | V1正确/1024 | V2正确/1024 |
|---|---|---:|---:|
| k11 | png8_control | 930 | 915 |
| k11 | jpeg_q90_444 | 897 | 887 |
| k11 | jpeg_q90_420 | 755 | 787 |
| k16 | png8_control | 729 | 711 |
| k16 | jpeg_q90_444 | 597 | 589 |
| k16 | jpeg_q90_420 | 333 | 393 |

错误归属总数（正样本认错用户与无水印图被接受，合计三种条件）：
- k11: V1 0 → V2 0。
- k16: V1 1 → V2 1。

未通过的预设开发检查：k11 jpeg_q90_444: 229 → 223, 保留率97.38%，低于98%要求

JPEG420恢复有所提升，PNG与JPEG444有所下降，因此不能把V2称为全面更好的正式版本。原4096源图是已观察过的回归队列，不是新独立测试；零观察错误也不代表真实错误概率为零。

审计通过：2400个fit批次、231个冻结参数/BN状态项、36864个双版本决策、36个结果单元格；V1发布ZIP与所有冻结输入未变。V2候选解压到新V1便携副本后，以独立配置加载通过，默认V1文件逐一hash不变。

CANDIDATE_BUNDLE.zip包含实际已训练的V2选中权重与独立inference_config.json。按LOAD_V2_IN_V1.md放到V1仓库的experiments/v2目录，通过显式--config调用。默认不带--config仍使用V1。

本目录保留所有六个额外检查点、optimizer/RNG、逐样本GT/logits/probabilities/决策、训练和选择历史。研究plan包含原Drive路径，属于私有实验记录；V1预训练发布包仍是之前整理的GitHub准备版本。

下一次恢复：打开RESTORE_V2_INFERENCE.ipynb，挂载同一Drive，恢复同一冻结环境后复核；该入口不训练。完整训练已完成，不要直接重复启动wouaf_extension_v2.py。

https://drive.google.com/drive/u/0/folders/1QsP-AbbpQsxVLBd_MJkGM2sRwR-deILH

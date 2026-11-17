# 结果、分母与解释

README第6节是最终的新独立内容验证，第7节是较早的已观察回归队列。严禁混合两者的百分比计算提升，也不能把所有压缩视图相加后称为独立样本。

## 最终验证证据

`reports/independent/summary.json` 与 `paired_recovery_statistics.csv`：每码每条件1024张正样本。主k11 JPEG420正确759→787，新增42、丢失14；辅k16 342→383，新增47、丢失6。PNG与444逐样本完全相同。正确、拒绝、错误归属拆分见 `recovery_and_false_attribution_cells.csv`。两类负样本各1024/条件/码；它们的accept一律算错误归属。

`predictions_V1/V3_k11/k16.npz` 保存真实GT、logits、probabilities、pred_bits、presence、候选、接收/拒绝、image/sample/scene id。`bit_error_characterization.json` 保存Hamming计数、真实BER、P(E=k)、每bit频率与相关矩阵；零方差bit的相关系数明示设为0。`analysis/recompute.py` 从这些文件重算，默认不碰测试图像和GPU。

CI按64基础场景整块配对bootstrap10000次、seed410930000；同块16个prompt/seed变体一起保留。McNemar p值为辅助描述，未作多重性校正。不能把零观察误归属称为概率为零，也不能由这一个压缩质量推断任意JPEG鲁棒性。

## V1/V2历史与失败尝试

`reports/v1/` 保存2400步真实训练、selection、calibration、逐图预测及初始版本对照；`reports/v2/` 保存追加2400步、选择extra800、固定规则核验和已观察回归表。V2的JPEG420改善伴随PNG/444损失，单独发布不满足保留目标。

`reports/v3/` 保存最终路由的固定policy检查、回归和实际文件头控制；`reports/v3_single_model_attempts/` 保留额外1600步与融合未达标的记录。失败训练选回step0等于V1，不是新模型提升。最终V3的有效提升来自按真实文件格式组合原两份权重。

保存原记录的JSON可能经过个人路径清理；`reports/provenance.json`同时记录原文件SHA与发布文件SHA，数值保持。原始未经清理研究目录仍保留在Drive。原有packaging helper的Path类型修正属于归档修复，没有改变预测、数据、统计或权重，相关记录保留。

## 与早期项目的关系

baseline固定3000 optimizer steps；“5000”是历史每组评估样本数。原baseline bit accuracy/CLIPScore和本系统accepted exact-user recovery是不同指标。历史干净条件k21/k16/k11的65.22%/82.64%/96.36%来自不同提取器与协议，不能直接视为此版本的同协议对照。报告正文应同时说明容量、拒绝和错误归属，不仅给一个最高恢复率。

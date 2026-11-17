# V2 扩展训练：完整码字目标＋JPEG420恢复

未通过全部预先固定的回归候选要求；V1仍是保留发布版本，V2结果仅作为实验记录。

父版本：<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/negative_aware_multitask_20261029T020157Z
父模型SHA256：39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb

此目录是独立实验版本，不修改之前GitHub准备发布包。模型为同一双头架构，selected_multitask.pth为选中权重；plan/protocol_lock记录预先固定的配方和规则，training_history/selection_history记录过程，last_resumable_checkpoint包含optimizer/RNG。

目标：在原双头loss基础上加入完整BCH码本交叉熵（除以32），并使用当前V1作为fit-only教师。真实图像条件按PNG/JPEG420/JPEG444/JPEG420/JPEG420循环；分组LR为1e-6/2.5e-6/5e-6/2e-5。2,400追加步，100 warmup + cosine；BN统计和前层冻结。

只有selection同时保持k11>=98%、k16>=95%的每条件正确数且两码都无坏接受时，才按JPEG420正确数等固定顺序选择；父模型也作为step0备选。BCH/bit/presence门槛全程固定，测试不用于选点/调门槛。calibration角色已在历史研究使用，因此此次也不是全新独立数据。

当前回归表见RESULTS_zh.html、regression_cells.csv；逐样本GT、logits、bits、presence和决策见regression_v2_k*.npz。观察到零错不等于真实零风险。未自动替换发布模型；需要新队列进一步验证泛化。

# V2 extension: full-codeword objective and JPEG420 recovery

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

What happened: further training helped JPEG420, but this candidate did not keep every required PNG/444 result. It was therefore not chosen as a replacement for V1 at that stage. Its trained JPEG420 specialist was later used as one component of the final V3. This folder records that earlier experiment; ordinary users can choose V3 in the app and skip these training details.

### Experiment details

Historical experiment record: this candidate failed the full predeclared retention requirements, so V1 remained the retained release at that stage. The current product defaults to final format-routed V3; V2 is its optional JPEG420 expert/control. No historical outcome is changed here.

Parent: `<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/negative_aware_multitask_20261029T020157Z`.
Parent model SHA256: `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb`.

This independent experiment did not overwrite the previous GitHub preparation package. It uses the same dual-head architecture; `selected_multitask.pth` was the selected checkpoint. Plan/protocol locks record the fixed recipe/rules, training/selection histories record progress, and the last resumable checkpoint retains optimizer/RNG state.

The objective adds full BCH-codebook cross-entropy divided by32 to the dual-head loss, with V1 as a fit-only teacher. Actual view conditions cycle PNG/JPEG420/JPEG444/JPEG420/JPEG420. Group learning rates are1e-6/2.5e-6/5e-6/2e-5. Training adds2400 steps with100-step warmup and cosine scheduling; BN statistics and earlier layers are frozen.

Selection requires at least98% of V1's per-condition correct count for k11,95% for k16, and no bad acceptance for either code. Candidates are ordered by the fixed JPEG420 criteria, with the parent as step0 fallback. BCH/bit/presence thresholds never change; tests are not used for selection/tuning. Calibration roles had been observed historically, so this is not fresh independent data.

Regression tables are in `RESULTS_zh.html`/`regression_cells.csv`; per-image GT/logits/bits/presence/decisions are in `regression_v2_k*.npz`. Seeing no mistakes in this sample does not mean the system can never make a mistake. This experiment did not automatically replace the release; later independent content validation addresses generalization.

---

<a name="chinese"></a>

## 中文

这轮做了什么：继续训练以改善JPEG420，但候选没有同时保住所有要求的PNG/444结果，因此当时没有整体替换V1。后来最终V3使用了其中训练好的JPEG420专家组件。这里保留的是早期实验记录；普通用户在界面选择V3即可，不需要照着下面重新训练。

### 实验详情

历史实验记录；当前成品默认版本见仓库根README中的最终V3。以下保留该阶段的方法和结果。

## V2 扩展训练：完整码字目标＋JPEG420恢复

未通过全部预先固定的回归候选要求；V1仍是保留发布版本，V2结果仅作为实验记录。

父版本：<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/negative_aware_multitask_20261029T020157Z
父模型SHA256：39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb

此目录是独立实验版本，不修改之前GitHub准备发布包。模型为同一双头架构，selected_multitask.pth为选中权重；plan/protocol_lock记录预先固定的配方和规则，training_history/selection_history记录过程，last_resumable_checkpoint包含optimizer/RNG。

目标：在原双头loss基础上加入完整BCH码本交叉熵（除以32），并使用当前V1作为fit-only教师。真实图像条件按PNG/JPEG420/JPEG444/JPEG420/JPEG420循环；分组LR为1e-6/2.5e-6/5e-6/2e-5。2,400追加步，100 warmup + cosine；BN统计和前层冻结。

只有selection同时保持k11>=98%、k16>=95%的每条件正确数且两码都无坏接受时，才按JPEG420正确数等固定顺序选择；父模型也作为step0备选。BCH/bit/presence门槛全程固定，测试不用于选点/调门槛。calibration角色已在历史研究使用，因此此次也不是全新独立数据。

当前回归表见RESULTS_zh.html、regression_cells.csv；逐样本GT、logits、bits、presence和决策见regression_v2_k*.npz。观察到零错不等于真实零风险。未自动替换发布模型；需要新队列进一步验证泛化。

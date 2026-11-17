# V3 supplementary interpolation

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

Historical control: interpolation failed all strict goals; V1 remained the retained release at that stage. This is not the current final V3 system.

This supplement was proposed after observing a one-PNG loss on the training selection set and was not in the original training protocol. The interpolation grid was locked before candidate inference; only selection roles chose the magnitude. Calibration was checked once and the old4096-source cohort served as regression, not a fresh independent test. Zero observed error is not zero population risk.

Source extra step:800. Layer4/bit-head mixing:(.5,.5). No new optimizer updates.

k11 V1 correct counts PNG/444/420:930/897/755; interpolated V3:931/895/756. False attribution0->0.

k16 V1 correct counts PNG/444/420:729/597/333; interpolated V3:723/598/335. False attribution1->1.

---

<a name="chinese"></a>

## 中文

历史实验记录；当前成品默认版本见仓库根README中的最终V3。以下保留该阶段的方法和结果。

## V3补充阶段：小幅权重融合

未达到全部严格目标，仅作实验记录；V1发布版保留。

本补充阶段是在看到训练选择集的PNG一张损失后提出；它不是原始训练协议的一部分。融合网格在任何融合候选推理前锁定，只用selection角色选幅度。calibration只核验一次，旧4096源图只作回归，不宣称新独立测试。零观察错不代表总体零风险。

源额外步数：800；layer4/bit-head融合比例：(0.5, 0.5)。无新增优化器更新。

k11 V1: {'png8_control': 930, 'jpeg_q90_444': 897, 'jpeg_q90_420': 755}，V3: {'png8_control': 931, 'jpeg_q90_444': 895, 'jpeg_q90_420': 756}；错误归属0→0。

k16 V1: {'png8_control': 729, 'jpeg_q90_444': 597, 'jpeg_q90_420': 333}，V3: {'png8_control': 723, 'jpeg_q90_444': 598, 'jpeg_q90_420': 335}；错误归属1→1。

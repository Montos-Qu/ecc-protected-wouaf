# V3 single-model attempt: three-view training and clean-view gradient retention

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

Historical failed candidate: it did not satisfy every fixed retention requirement, so V1 remained the retained release at that stage. This is not the final format-routed V3 product.

The model starts from V1 and trains only layer4 and the bit head; earlier layers, BN statistics and the presence head are frozen. Every step uses the same12 fit images as PNG/JPEG444/JPEG420 views. V1 teaches PNG/444 with logit and correct-bit-margin retention. V2 supplies JPEG420 teacher outputs only on fit images where its decoding is correct and GT BCE is lower. Gradient projection makes the first-order SGD update nonconflicting with both clean-view objectives. This does not guarantee unseen-image retention; development/regression checks remain necessary.

SGD uses no momentum or weight decay, LR.002,100-step warmup, cosine decay to10%, gradient norm clipping1,1600 steps, and selection every400 steps including step0. Both codes must have zero false attribution, PNG/444 correct counts no lower than V1, and420 no lower than V1. Ranking prioritizes k11 JPEG420, total correct count, then analogous k16 counts. Calibration/regression also require every-condition retention and strict primary-code420 improvement; rejection thresholds are fixed.

Old development and4096-source regression cohorts had been observed before and are not fresh independent tests. Fit/selection/calibration roles remain isolated. V1/V2 are read-only references; this separate experiment does not automatically replace the release. Checkpoints, GT, logits, probabilities, decisions, optimizer/RNG, training and selection records are retained.

---

<a name="chinese"></a>

## 中文

历史实验记录；当前成品默认版本见仓库根README中的最终V3。以下保留该阶段的方法和结果。

## V3：三视图训练与PNG/JPEG444梯度保留

未通过全部预先固定的回归候选要求；V1仍是保留发布版本，V3结果仅作为实验记录。

从V1初始化，只训练layer4与bit head，前层、BN统计及presence head全部冻结。每步同一12张fit图像同时使用PNG、JPEG444、JPEG420三个视图；V1作为PNG/444教师，加logit与正确bit margin保留。V2只在fit图像上提供正确解码且GT BCE更低的420教师输出。梯度投影使SGD更新与两份干净视图训练目标一阶不冲突；这不保证未见图像不退化，仍以开发与回归结果核验。

SGD无momentum、无weight decay、统一LR0.002，100步warmup和cosine到10%，梯度范数裁剪1，共1600步，每400步选点，起点也作为备选。选点要求两码无错误归属、PNG/444正确数均不低于V1、420不低于V1，再按k11 JPEG420正确数、总正确数、k16同类计数排序。calibration与回归也要求每条件不退化，主码420严格改善，拒绝门槛不调整。

旧开发和4096源图回归队列均已在历史研究观察，不能称为全新独立测试。所有fit/selection/calibration角色保持隔离。V1与V2仅作为只读参考；本目录是单独V3版本，不自动替换发布包。权重、GT、logits、概率、决策、优化器/RNG、训练和选点记录均保留。

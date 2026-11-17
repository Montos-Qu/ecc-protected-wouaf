# 独立训练目录与复现边界

预训练V3已经包含两份真实权重。训练和未训练模型均放在 `training/` / 新 `runs/`；默认推理不触发这些代码。

## V1

从归档+300单头WOUAF提取器初始化backbone/bit head，新建presence head。开发图4456张；内容/重复用户连接关系约束角色隔离，fit=folds0/1，selection=fold2，calibration=fold3。便携 `recipe.dataset()` 核对逐图SHA、item顺序、同内容与同用户的角色唯一性。

2400步，batch24：4个k11正样本、4个k16正样本、8个配对同源负样本、4个额外生成负样本、4个自然负样本。PNG/444/420/420轮换。只训练layer3、layer4、bit head、presence head；BN统计冻结，presence内部dropout用于训练。AdamW，weight_decay1e-4；四组LR分别2e-6/5e-6/1e-5/1e-4；100步warmup+cosine；grad clip1。

loss = 2×bit BCE + presence BCE + .05×负样本bit不确定性 + .05×原初始化bit-logit MSE + .2×正/配对负presence margin。每400步做selection，按预定保留/正确数排序。选点锁定后才在calibration角色拟合presence拒绝规则；最终发布两码presence门槛都是0。测试数据不参与选点或门槛拟合。

入口：`training/train.py`、`calibrate.py`。`--dry-run`只检查4456图及12个初始化参考输出，不创建optimizer或训练目录。`--resume`使用当前新run的optimizer和RNG断点；所有输出需独立命名。发布V1选中2400步，hash见configs。

## V2

入口 `training/train_v2.py`；从已训练V1初始化，seed20261030，追加2400步，保留原BCH/bit/presence门槛。PNG/420/444/420/420轮换。AdamW的四组LR降为1e-6/2.5e-6/5e-6/2e-5；其余warmup/cosine/clip/BN冻结沿用。教师为实际父V1，仅对fit角色生成输出。

在bit/presence/负样本中性/配对loss上加入两个完整BCH码本的CE：2048或65536个合法32-bit码字，分数为logits@codebook.T，CE除32后两码平均；同时保留父V1 bit/presence。bit教师MSE权重420为.01、其他为.1；presence保留.01。不是只训练当前GT白名单。

每400步仅用selection评估；候选要求两码bad=0，k11每条件正确保留≥98%、k16≥95%，再按420与总体正确数排序。父V1也是step0备选。历史最终选extra800；后续calibration是同一冻结规则的核验，未重调；旧回归虽420更好，但PNG/444保留失败，故V2最终是专家。

`--dry-run`检查4456图、两个完整码本、12个原V1 fit参考输出和CE等价式；零optimizer更新。`--resume`恢复新run的权重、optimizer和RNG。新run可用 `training/infer_single.py --config runs/NEW/inference_config.json --image IMAGE --output outputs/NEW.json` 显式推理；不会自动改发布V3配置。

## 最终V3与原始训练来源

最终V3不再优化参数：实际文件头JPEG420选V2，其他选V1，门槛相同。单模型V3额外1600步与权重融合未达全部条件，相关原始代码/报告在research_reference和reports的明确失败尝试目录，不当作发布模型。

`research_reference/` 保存经过个人路径清理的原研究脚本及计划，供审阅原优化过程；它们保留历史的路径结构，不能把这些参考脚本视为便携CLI。可执行的当前提取器入口是本目录 `train.py/train_v2.py/calibrate.py`。完整WOUAF生成器baseline原训练源和断点在 `baseline_generator` 资产的 `code/` 与 `checkpoint-3000/`；未宣称其所有上游从头训练分支在当前环境一键通过。

本次打包只运行dry-run和已保存数值回归，不重新执行完整2400+2400步，不承诺便携重训权重逐字节复现。新训练必须另存、独立选择/校准/验证；已看过的最终独立测试不能反复调参后继续称为未见测试。

随机架构factory：`training/create_untrained_model.py --output runs/random/untrained.pth`，标记trained:false；其模型结构可供自行研究，但随机训练2400步不是本项目从+300初始化的配方。

# V3 从这里开始

推荐查看 format_aware_candidate/README_zh.md 与 RESULTS_zh.html。最终可运行候选包是 format_aware_candidate/V3_FORMAT_ROUTED_DELIVERY.zip；恢复入口为该目录的 RESTORE_V3_FORMAT_ROUTER.ipynb。

V3采用实际文件头自动选择V1或V2专家，属于双提取器推理组合。追加1600步的单模型与权重融合未达联合保留要求，旧V3_DELIVERY.zip仅为该失败实验的记录。两个阶段明确分开，不覆盖V1/V2。

当前格式范围内已保住PNG/444并提升420。 旧队列仅作回归，需新独立验证。

勘误：早期单模型包LOAD_V3_DELIVERY.md对原infer.py --condition的描述错误；原入口JPEG条件会额外重编码，新auto入口只读取真实文件，不重编码。

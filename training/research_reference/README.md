# Historical research source

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

This is the project's earlier experiment code, kept so readers can check how the results were obtained. It is not the app launcher, and some historical paths need adapting. For normal use, follow the root README. For new training, use the supported files in training/ and the training guide.

### Experiment details

This directory preserves original recipes and audit provenance. Personal paths were sanitized; original and distributed hashes are recorded in `reports/provenance.json`. Historical directory structures need adaptation and are not the default portable product entry. Executable custom-training entry points are `training/train.py`, `training/train_v2.py` and `training/calibrate.py`.

Final V3 routes between the two immutable trained extractors using the file header. Failed single-model and interpolation experiments are historical controls, not the final product. Normal consumers run `python wouaf.py setup` and `python wouaf.py predict`; these never invoke training.

---

<a name="chinese"></a>

## 中文

这里保存早期实验代码，方便查看结果是怎么得到的。它不是成品启动入口，历史路径可能需要调整。日常使用看根目录README；想自行训练时，使用training/中的正式入口和训练说明。

### 实验详情

此处保留原配方与审计来源，个人路径已清理；原SHA和发布SHA在reports/provenance.json。历史路径结构需要适配，非便携默认入口。可执行的自行训练入口是training/train.py、train_v2.py、calibrate.py。最终V3是两份原权重的文件头路由；单模型/融合失败记录不当作最终模型。

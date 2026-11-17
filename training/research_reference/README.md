# Historical research source

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

This directory preserves original recipes and audit provenance. Personal paths were sanitized; original and distributed hashes are recorded in `reports/provenance.json`. Historical directory structures need adaptation and are not the default portable product entry. Executable custom-training entry points are `training/train.py`, `training/train_v2.py` and `training/calibrate.py`.

Final V3 routes between the two immutable trained extractors using the file header. Failed single-model and interpolation experiments are historical controls, not the final product. Normal consumers run `python wouaf.py setup` and `python wouaf.py predict`; these never invoke training.

---

<a name="chinese"></a>

## 中文

此处保留原配方与审计来源，个人路径已清理；原SHA和发布SHA在reports/provenance.json。历史路径结构需要适配，非便携默认入口。可执行的自行训练入口是training/train.py、train_v2.py、calibrate.py。最终V3是两份原权重的文件头路由；单模型/融合失败记录不当作最终模型。

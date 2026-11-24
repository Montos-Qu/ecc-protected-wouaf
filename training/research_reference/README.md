# Historical research source

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

This is the project's earlier experiment code, kept so readers can check how the results were obtained. It is not the app launcher, and some historical paths need adapting. For normal use, follow the root README. For new training, use the supported files in training/ and the training guide.

### Experiment details

This directory preserves original recipes and audit provenance. Personal paths were sanitized; original and distributed hashes are recorded in `reports/provenance.json`. Historical directory structures need adaptation and are not the default portable product entry. Executable custom-training entry points are `training/train.py`, `training/train_v2.py` and `training/calibrate.py`.

Final V3 routes between the two immutable trained extractors using the file header. Failed single-model and interpolation experiments are historical controls, not the final product. Normal consumers run `python wouaf.py setup` and `python wouaf.py predict`; these never invoke training.

---

<a name="chinese"></a>

## ����

���ﱣ������ʵ����룬����鿴�������ô�õ��ġ������ǳ�Ʒ������ڣ���ʷ·��������Ҫ�������ճ�ʹ�ÿ���Ŀ¼README��������ѵ��ʱ��ʹ��training/�е���ʽ��ں�ѵ��˵����

### ʵ������

�˴�����ԭ�䷽�������Դ������·����������ԭSHA�ͷ���SHA��reports/provenance.json����ʷ·���ṹ��Ҫ���䣬�Ǳ�ЯĬ����ڡ���ִ�е�����ѵ�������training/train.py��train_v2.py��calibrate.py������V3������ԭȨ�ص��ļ�ͷ·�ɣ���ģ��/�ں�ʧ�ܼ�¼����������ģ�͡�

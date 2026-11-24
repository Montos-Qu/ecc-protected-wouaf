# Trained model files

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

These two`.pth` files contain what the models learned during training. They are already trained: users do not need to train them again.

From the project root, run `python wouaf.py setup`. It checks existing files and downloads missing ones automatically. The complete product ZIP already includes both weights; GitHub's source ZIP/clone downloads them during setup. Large weights are stored in GitHub Releases rather than ordinary Git history.

| Component | How it is used |
|---|---|
| V1, trained2400 steps | Balanced image reader; used by V1 and by V3 for PNG/JPEG444 |
| V2, selected at800 extra steps | JPEG420 specialist; used by V2 and by V3 for JPEG420 |

V3 chooses between these components from the actual image format. There is no missing third V3 file. Both use the original architecture and fixed decision thresholds.

For a weights-only download, run `python scripts/download_pretrained.py` from the root. `pretrained_manifest.json` records the official project Release URLs, sizes and SHA256 checksums. A matching file is reused. A conflicting file causes an error without being overwritten; the app never substitutes a random untrained model.

---

<a name="chinese"></a>

## ����

������`.pth`�ļ�������ģ��ѵ��ѧ���Ĳ�����Ҳ���ǡ�Ȩ�ء��������Ѿ�ѵ���ã�ʹ���߲�������ѵ����

����Ŀ��Ŀ¼���� `python wouaf.py setup` �����Զ���������ļ�������ȱʧȨ�ء�������ƷZIP�Ѿ���������Ȩ�أ�Դ��ZIP/��¡���������ʱ�Զ����ء���Ȩ�ر�����GitHub Releases�����Ž���ͨGit��ʷ��

| ��� | ��ôʹ�� |
|---|---|
| V1��ѵ��2400�� | ƽ����ȡ����V1ʹ�ã�V3����PNG/JPEG444ʱҲʹ�� |
| V2��ѡ��׷��800����Ȩ�� | JPEG420ר�ң�V2ʹ�ã�V3����JPEG420ʱҲʹ�� |

V3����ʵͼƬ��ʽѡ�����������ȱ���˵�����V3�ļ����������ԭģ�ͽṹ�͹̶��ж��ż���

���ֻ������Ȩ�أ��ڸ�Ŀ¼ִ�� `python scripts/download_pretrained.py`��`pretrained_manifest.json`������ĿRelease���ص�ַ����С��SHA256У��ֵ����ȷ�ļ�ֱ�Ӹ��ã������ļ������������ǣ����򲻻������δѵ��ģ�ʹ��档

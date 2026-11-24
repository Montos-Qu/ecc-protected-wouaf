# ECC-Protected WOUAF �� Pretrained V3

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

**A ready-to-use tool for reading WOUAF fingerprints from images. Download it, set it up, and use the trained V3 model. You do not need to train anything.**

If an image contains a recoverable WOUAF fingerprint, the app can report its user ID. If it cannot recover enough reliable information, it shows `REJECT` and explains why. This is an image-reading tool: it does not add fingerprints to your photos, detect every kind of watermark, or identify the author of any ordinary picture.

### 1. Start here: Windows web app

1. Open the [latest Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/latest) and download the **complete product ZIP** whose name ends in `local-3.0.6-final.zip`. Choose this file rather than GitHub's automatically generated source ZIP if you want weights and demo files included.
2. Extract the whole ZIP into a normal folder. Do not run files from inside the ZIP.
3. Double-click `start_ui.cmd`. The first run downloads Python and installs the required packages in this project folder. Keep the window open until setup finishes. Internet is needed for this first setup; you do not need to preinstall Python, use Colab, log into the author's Drive, or have an A100.
4. Your default browser opens WOUAF Studio. Upload a **native512x512 PNG or JPEG**, keep the default **V3**, and click **Identify fingerprint**.
5. Read the user ID or rejection reason. **Download result JSON** saves the full result. A copy is also kept in `outputs/ui/`.

To check that your installation works, upload `demo/images/png8_control/positive_k11/10000000.png`. With the default V3/k11 settings, the supplied positive example should return **user565**. An ordinary photo may be rejected; that alone is not an installation problem. A watermarked JPEG can also be rejected if its fingerprint is too damaged.

Use **Stop app** to stop the local service. Closing the browser tab alone leaves it running. Starting the same project again reopens its existing service; a different project copy uses another free local port if necessary. Images are processed on your computer, without a cloud upload. The page supports English and Chinese and includes a larger desktop layout. See the [UI guide](docs/LOCAL_UI.md).

### 2. What pictures work, and what do the results mean?

- Use original512x512 PNG/JPEG files, no larger than10MiB. The app does **not** resize, crop, or add borders. Resizing can damage the fingerprint even if the final file is512x512. See [why automatic resizing was removed](docs/IMAGE_SIZE_LIMITS.md).
- `ACCEPT`: the recovered fingerprint passed the model's fixed checks. `user_id` is the accepted ID.
- `REJECT`: the checks did not pass, so no user is attributed and `user_id` is `null`. This does not prove the image has no watermark. The page explains the first failed check, such as too many bit errors or insufficient evidence.
- The JSON includes the chosen version, the extractor actually used, image format, model-file checksum,32 predicted bits, raw model scores (`logits`), bit probabilities, and the rejection reason. Bit probabilities are not the probability that an identity is correct.

The measured accuracy applies to original512x512 RGB images, PNG and Pillow JPEG Q90 with4:4:4/4:2:0 sampling. Other qualities, editing, screenshots and attacks are not covered by those numbers. Converting a JPEG to PNG does not undo compression and also removes its original JPEG sampling information.

### 3. V3, V1 and V2: which should I choose?

| Choice | What it does | When to use it |
|---|---|---|
| **V3, default** | Uses V1 for PNG/JPEG444 and V2 for actual JPEG420 files | **Recommended final product** |
| V1 | Uses the balanced extractor for every image | Compare with the earlier balanced version |
| V2 | Uses the JPEG420 specialist for every image | Compare JPEG420 results; PNG/444 results may be worse |

V3 chooses by the actual file header, not its extension or a known answer. It stores two trained components and runs one per image. It does not need a separate third V3 weight file.

The default error-correcting code, BCH(31,11)+overall parity (`k11`), supports2048 user IDs. The optional `k16` setting supports65536 IDs but recovers fewer images in our tests. These are capacity choices, not different amounts of training. Start with V3 and k11.

### 4. Prefer commands?

Windows without Python can run `setup_windows.cmd` and then drag a picture onto `predict_windows.cmd`, or run:

```bat
predict_windows.cmd "YOUR_IMAGE.png" v3
predict_windows.cmd "YOUR_IMAGE.jpg" v1
predict_windows.cmd "YOUR_IMAGE.jpg" v2
```

If you already have Python3.9+, clone/download the repository, open a terminal in its folder, and run:

```bash
python wouaf.py setup
python wouaf.py ui
python wouaf.py predict --image YOUR_IMAGE.png --output outputs/prediction.json
python wouaf.py predict --image YOUR_IMAGE.jpg --version v1
python wouaf.py predict --image YOUR_IMAGE.jpg --version v2
python wouaf.py predict --image YOUR_IMAGE.png --code 16
python wouaf.py versions
python wouaf.py verify
```

`setup` installs the exact locked package versions automatically; no manual comparison with a package list is needed. It also downloads missing trained weights and support files, checks their hashes, and tests the model. The complete product ZIP already includes those project files; the source ZIP/clone obtains them during setup. Setup does not train a model or change your system Python. After setup, inference works locally. The [public notebook](01_PUBLIC_V3_REPRODUCE.ipynb) is an optional alternative without mounting the author's Drive.

### 5. Computer requirements and environment

Supported installation profiles: Windows10/11 x64 and x86_64 Linux with glibc2.28+. An NVIDIA GPU can accelerate inference; CPU mode also works on a computer with AMD graphics. AMD GPU acceleration, macOS and ARM are not included in these profiles.

```bash
python wouaf.py setup --device cpu
python wouaf.py setup --device cuda
python wouaf.py predict --image YOUR_IMAGE.png --device cpu
python wouaf.py ui --no-browser
```

The project manages Python3.12.12, PyTorch2.7.1, torchvision0.22.1, NumPy1.26.4, Pillow12.3.0 and SciPy1.15.3. All dependencies are pinned and checked against hashes. Automatic setup tries usable NVIDIA CUDA, otherwise CPU. It selects official CUDA11.8 or12.8 packages for the GPU generation; RTX50/compute capability10+ selects12.8. `--cuda-backend cu118` or `--cuda-backend cu128` overrides that choice. A compatible NVIDIA driver is required; setup does not install drivers. See [runtime details](docs/PORTABLE_RUNTIME.md).

The components were trained on an A100. **Ordinary users do not need an A100 or the original training environment.** Local mode uses the same weights and decision rules with its own fixed package versions. Windows CPU was tested against saved A100 outputs on48 native fixtures: no changed fingerprint bits or accepted IDs; maximum raw-score difference0.0000362396240234375. This finite check does not prove identical results for every image or GPU. NVIDIA CUDA and Linux paths were not hardware-tested in this Windows/AMD session.

For researchers who need the historical A100 environment, use `python wouaf.py setup --runtime reference` and `python wouaf.py predict --runtime reference --image YOUR_IMAGE.png`. That separate mode checks Linux Python3.10.21, torch1.12.1+cu116, A100-SXM4-80GB and driver580.82.07. See [reference environment](docs/ENVIRONMENT.md). Normal setup does not download that archive.

### 6. Results and optional research downloads

On a new independent image set, with1024 watermarked images per code and format, the share correctly recovered and accepted was:

| Capacity | PNG: V1 �� V3 | JPEG444 Q90: V1 �� V3 | JPEG420 Q90: V1 �� V3 |
|---|---:|---:|---:|
| k11, default |90.14% ��90.14%|87.60% ��87.60%|**74.12% ��76.86%**|
| k16, larger capacity |67.68% ��67.68%|56.35% ��56.35%|**33.40% ��37.40%**|

V3 improves JPEG420 while keeping PNG/444 outputs identical to V1 for this test. The paired scene-cluster95% interval for the420 gain is +1.37 to +4.10 percentage points for k11 and +2.64 to +5.47 for k16. No false user attribution was observed in this finite study; that is not a guarantee of zero risk. The original final table does not claim a full new-set V2 evaluation across every format. Earlier V1/V2 comparisons, failed attempts and limits are kept in [results](docs/RESULTS.md) and the [research workflow](docs/RESEARCH_REPRODUCTION_zh.md).

**You do not need the following large research downloads for normal use.** They are public, read-only downloads for replaying experiments. The old environment has3 parts; the six image groups are for full independent evaluation:

| Public asset group | Download | Compressed size |
|---|---|---:|
| Frozen runtime: three parts | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| Native JPEG420 views | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |
| Native JPEG444 views | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| Generated negative PNGs | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| Natural negative PNGs | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| Positive k11 PNGs | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| Positive k16 PNGs | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |

Exact file URLs, sizes and SHA256 checksums are in [public_assets.json](public_assets.json). The environment parts total2,319,751,708 bytes; environment plus images total4,829,967,179 bytes, with extra disk space needed for extraction. Optional downloads go to `offline_assets/`, or `--asset-root PATH`.

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
```

See [full reproduction steps](docs/PUBLIC_REPRODUCTION.md) to use saved per-image predictions or rerun image inference. The original generator, training data and5000 latents are separate research assets; the trained app and this fixed-image evaluation do not need them. `00_PRETRAINED_V3_DEMO.ipynb` is the historical author-Drive restore notebook, not the ordinary public setup.

### 7. Files, trained weights and optional training

| Folder/file | Purpose |
|---|---|
| `start_ui.cmd`, `wouaf.py` | Normal app and command-line entry points |
| `models/` | Already-trained V1/V2 component weights; V3 uses both |
| `configs/`, `ecc_wouaf/`, `ecc/` | Version choices, image reading, bit recovery and fixed acceptance rules |
| `environment/portable/` | Local environment installer and dependency locks |
| `environment/locked/` | Historical reference environment support |
| `training/` | Optional code for people who want to train a different model |
| `reports/`, `docs/` | Experiments, results, limits and reproduction guides |
| `outputs/` | Your local results and installation records |

**Weights** are the parameters learned during training, saved as`.pth` files. The provided weights make this a trained product. If they are missing, setup downloads them; it does not ask users to retrain or silently use a random model. The original files stay available in the [v3.0.0 model Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/tag/v3.0.0):

| File | SHA256 checksum |
|---|---|
| `models/v1_negative_aware_2400.pth` | `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb` |
| `models/v2_jpeg420_expert_extra800.pth` | `7a5f5b5b3149dd4ebd2837e3a36367fbb3cda4691935ca017c24fd403b48827d` |

Training is separate and optional; see [training instructions](docs/TRAINING.md). Normal setup, the UI and prediction commands never start it. Large files use GitHub Releases/public Drive instead of ordinary Git history.

### 8. Original WOUAF and credits

This Applied Project builds on [kylemin's original WOUAF repository](https://github.com/kylemin/WOUAF), accompanying the CVPR2024 work. Our additions cover error correction, parity and rejection tests, V1/V2/V3 comparisons, local setup and the web UI. Credit for the original method and implementation remains with its authors. Use the [upstream citation](https://github.com/kylemin/WOUAF#citation) for academic references; see [references](docs/REFERENCES.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Public downloads do not replace the original model, image or library license terms.

---

<a name="chinese"></a>

## ����

**����һ�����غ����ʹ�õ� WOUAF ͼƬָ��ʶ�𹤾ߡ�Ĭ�ϼ���ѵ���õ� V3����������ѵ����**

���ͼƬ�ﱣ�����ָܻ��� WOUAF ָ�ƣ�����������Ӧ�û�ID����Ϣ����ʱ��ʾ `REJECT`����˵��ԭ���������ȡ����ָ�ƣ��������Ƭ����ָ�ƣ�Ҳ����ͨ��ˮӡ�����������ʶ��������ͨ��Ƭ�����ߡ�

### 1. Windows �û������￪ʼ

1. ��[���� Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/latest)������������ `local-3.0.6-final.zip` ��β��**������Ʒ ZIP**�����Ѿ�����ѵ��Ȩ�غ���ʾ�ļ���GitHub �Զ����ɵ�Դ�� ZIP ��������Щ���ļ���
2. ������ ZIP ��ѹ����ͨ�ļ��У���Ҫ��ѹ������ֱ�����С�
3. ˫�� `start_ui.cmd`���״λ��Զ����� Python ����װ���������л����ļ���������ĿĿ¼�������ý������״���Ҫ����������Ԥװ Python������ Colab�����õ�¼���ߵ� Drive��Ҳ���� A100��
4. Ĭ���������� WOUAF Studio���ϴ�**ԭʼ512��512�� PNG/JPEG**������Ĭ�� **V3**�����**��ʼʶ��ָ��**��
5. �鿴�û�ID��ܾ�ԭ�򡣵��**�������� JSON ���**�ɱ�����ϸ���������Ҳ���� `outputs/ui/` ��һ�ݸ�����

����ȷ�ϰ�װ�����������ϴ� `demo/images/png8_control/positive_k11/10000000.png`������Ĭ��V3/k11���渽�����Ŵ�ָ��ʾ��Ӧ��ʾ**�û�565**����ͨ��Ƭ���ܾ���������װʧ�ܣ���ָ��JPEGѹ����Ҳ������Ϊָ����������ܾ���

���**�رշ���**�Ż�ֹͣ��̨��ֻ���������ǩҳ����ֹͣ���ٴ�����ͬһĿ¼������з�����һ����Ŀ���������˿�ռ�����Զ������ж˿ڡ�ͼƬֻ�ڱ������������ϴ��ƶˡ�����֧����Ӣ�ĺͽϴ�������ֺţ����[��ҳ����˵��](docs/LOCAL_UI.md)��

### 2. ���ϴ�ʲôͼƬ�������ô����

- ʹ��ԭʼ512��512 PNG/JPEG������10MiB������**�����š����ü���������**���ֶ��ĳ�512Ҳ���ָܻ�����ʱ��ʧ��ָ����Ϣ����[Ϊʲôȡ���Զ�����](docs/IMAGE_SIZE_LIMITS.md)��
- `ACCEPT`���ָ���ָ��ͨ���˹̶���飬`user_id` �Ǳ����ܵ��û�ID��
- `REJECT`��û��ͨ����飬��˲��������κ��û���`user_id` Ϊ `null`���ⲻ��֤��ͼƬû��ˮӡ��ҳ����������ʧ�ܵĹ����������λ̫�࣬��֤�ݲ��㡣
- JSON������ѡ�汾��ʵ��ʹ�õ���ȡ����ͼƬ��ʽ��ģ���ļ�У��ֵ��32λԤ��ָ�ơ�ԭʼģ�ͷ�����`logits`����ÿλ���ʺ;ܾ�ԭ��ÿλ���ʲ����ڡ��û�������ȷ�ĸ��ʡ���

ʵ��׼ȷ�ʶ�Ӧԭʼ512��512 RGB��PNG���Լ�Pillow Q90��JPEG444/JPEG420������ѹ���������༭����ͼ�򹥻�����������Щ׼ȷ�����JPEGת��ΪPNG�Ȳ�������ѹ�����ˣ�Ҳ�ᶪʧԭJPEG������Ϣ��

### 3. V3��V1��V2 ��ôѡ��

| ѡ�� | ��ʲô | �ʺ���; |
|---|---|---|
| **V3��Ĭ��** | PNG/JPEG444��V1����ʵJPEG420��V2 | **���ճ�Ʒ���ճ��Ƽ�** |
| V1 | ����ͼƬ����ƽ����ȡ�� | ������ƽ��汾�Ա� |
| V2 | ����ͼƬ����JPEG420ר�� | �Ա�JPEG420Ч����PNG/444�����½� |

V3��ȡ��ʵ�ļ�ͷ��ѡ��ģ�ͣ�������չ������֪�𰸡���Ŀ��������ѵ�������ÿ��ͼƬֻ��������һ�ݣ�����Ҫ������������V3Ȩ���ļ���

Ĭ�Ͼ�����ʽBCH(31,11)+������żУ�飨`k11`��֧��2048���û�ID����ѡ `k16` ֧��65536��ID������ʵ���еĳɹ��ָ��ʽϵ͡������û�������ȡ�ᣬ����ѵ������������һ�㱣��V3��k11���ɡ�

### 4. ���������У�

û��Python��Windows�û��������� `setup_windows.cmd`���ٰ�ͼƬ�ϵ� `predict_windows.cmd` �ϣ�Ҳ��ִ�У�

```bat
predict_windows.cmd "YOUR_IMAGE.png" v3
predict_windows.cmd "YOUR_IMAGE.jpg" v1
predict_windows.cmd "YOUR_IMAGE.jpg" v2
```

�Ѿ���װPython3.9+���û������Կ�¡�����زֿ⣬����ĿĿ¼���ն�ִ�У�

```bash
python wouaf.py setup
python wouaf.py ui
python wouaf.py predict --image YOUR_IMAGE.png --output outputs/prediction.json
python wouaf.py predict --image YOUR_IMAGE.jpg --version v1
python wouaf.py predict --image YOUR_IMAGE.jpg --version v2
python wouaf.py predict --image YOUR_IMAGE.png --code 16
python wouaf.py versions
python wouaf.py verify
```

`setup` �Զ��������汾��װ���������ö����嵥������ء������Ჹ��ѵ��Ȩ�غ�֧���ļ���У���ļ��Ƿ�������������ģ�͡�������ƷZIP�Ѱ�����Щ��Ŀ�ļ���Դ��ZIP/��¡����ڰ�װʱ���롣��װ����ѵ��ģ�ͣ�Ҳ���޸�ϵͳPython��������ɺ��ڱ���������[���� notebook](01_PUBLIC_V3_REPRODUCE.ipynb)ֻ����һ�ֿ�ѡ��ڣ�����Ҫ��������Drive��

### 5. ����Ҫ���뻷������

Ŀǰ�ṩWindows10/11 x64��x86_64 Linux��glibc2.28+����װ�������м���NVIDIA�Կ����Լ��٣���AMD�Կ��ĵ���Ҳ��ʹ��CPUģʽ�����治����AMD�Կ����١�macOS��ARM���з�����

```bash
python wouaf.py setup --device cpu
python wouaf.py setup --device cuda
python wouaf.py predict --image YOUR_IMAGE.png --device cpu
python wouaf.py ui --no-browser
```

�����Զ�����Python3.12.12��PyTorch2.7.1��torchvision0.22.1��NumPy1.26.4��Pillow12.3.0��SciPy1.15.3�������������й̶��汾���ļ�У�顣�Զ���������ʹ�ÿɹ�����NVIDIA CUDA���������CPU�����Կ�����ѡ��ٷ�CUDA11.8/12.8����RTX50/compute capability10+ʹ��12.8��Ҳ���� `--cuda-backend cu118` �� `--cuda-backend cu128` ָ�����Կ�������Ҫ�û����м��ݰ汾�����򲻰�װ���������[�������л���](docs/PORTABLE_RUNTIME.md)��

�������A100��ѵ���ģ�**��ͨʹ�ò���ҪA100��Ҳ����Ҫ�ؽ�ѵ������**������ģʽ����ԭȨ�غ��жϹ���ʹ�õ����̶��������汾��Windows CPU���뱣���A100����Ա�48��ԭʼͼƬ��ָ��λ���ѽ����û�ID���ޱ仯�����ԭʼ������0.0000362396240234375�����޲��Բ���֤������ͼƬ�������Կ�����ȫһ�¡�����Windows/AMD����û��ʵ��NVIDIA CUDA��LinuxӲ����

ֻ����Ҫ������ʷA100ʵ����о��ߣ���ʹ�� `python wouaf.py setup --runtime reference` �� `python wouaf.py predict --runtime reference --image YOUR_IMAGE.png`����ģʽ���Linux Python3.10.21��torch1.12.1+cu116��A100-SXM4-80GB��driver580.82.07����[�ο�����˵��](docs/ENVIRONMENT.md)����ͨ��װ����������ݾɻ����鵵��

### 6. ʵ�������ѡ�о�����

��һ���µĶ���ͼƬ�ϣ�ÿ�ֱ��롢ÿ�ָ�ʽ��1024�Ŵ�ָ��ͼƬ���ɹ��ָ���ȷ�û���ͨ�����ı���Ϊ��

| �������� | PNG��V1 �� V3 | JPEG444 Q90��V1 �� V3 | JPEG420 Q90��V1 �� V3 |
|---|---:|---:|---:|
| k11��Ĭ�� |90.14% ��90.14%|87.60% ��87.60%|**74.12% ��76.86%**|
| k16���ϴ����� |67.68% ��67.68%|56.35% ��56.35%|**33.40% ��37.40%**|

��β�����V3�����JPEG420Ч����PNG/444����ͼ�����V1��ȫһ�¡���ͬԴ����������㣬420������95%�������Ϊ��k11 +1.37��+4.10���ٷֵ㣬k16 +2.64��+5.47���ٷֵ㡣�������޲���δ�۲쵽�����û���������������ʵ����Ϊ�㡣ԭ���ձ�û���������V2��������ȫ����ʽ�ϵ�ȫ�����ԡ�����V1/V2���ա�ʧ�ܳ��Ժ����Ʊ�����[���˵��](docs/RESULTS.md)��[�о�����](docs/RESEARCH_REPRODUCTION_zh.md)��

**�ճ�ʹ�ò�������������Щ�����о����ݡ�** ���ǿ�������ֻ�����أ�������ʵ�飻�ɻ�����3����Ƭ��6��ͼƬ������������������

| �����ļ��� | �������� | ���ش�С |
|---|---|---:|
| ���� Python��PyTorch ��ȫ��������3 ����Ƭ�� | [frozen_environment](https://drive.google.com/drive/u/0/folders/16Lve4kE37wyv8aooVufue7Sga5GYCXRt) | 2.16 GiB |
| k11 ˮӡ PNG ������ | [independent_source_pngs_positive_k11](https://drive.google.com/drive/u/0/folders/1QKEYbcB803RMJ51x5Mzb36eZ6PI8LuFA) | 0.43 GiB |
| k16 ˮӡ PNG ������ | [independent_source_pngs_positive_k16](https://drive.google.com/drive/u/0/folders/1NUgPI13HwKFWIoLI-i0dHWBi7kl9r0uO) | 0.43 GiB |
| ͬԴ��ˮӡ PNG ������ | [independent_source_pngs_generated_negative](https://drive.google.com/drive/u/0/folders/1MAbvrbca6pEMSOhdRQ7itg_piP0zVbh1) | 0.44 GiB |
| ��Ȼͼ PNG ������ | [independent_source_pngs_natural_negative](https://drive.google.com/drive/u/0/folders/1--F3b4k6_mGp-yFE1qAqjc0OjxR54QeL) | 0.38 GiB |
| ��ʵ JPEG444 Q90 ��֤��ͼ | [independent_encoded_jpeg_q90_444](https://drive.google.com/drive/u/0/folders/1BHrB2RYjG6jKitZDuo1lAM88K8tTZ_BA) | 0.37 GiB |
| ��ʵ JPEG420 Q90 ��֤��ͼ | [independent_encoded_jpeg_q90_420](https://drive.google.com/drive/u/0/folders/1MmT0jjIyqGQ3P0iao8vbs7uyJnFhzMJ1) | 0.29 GiB |

׼ȷ���ص�ַ����С��SHA256У��ֵ�� [public_assets.json](public_assets.json)��������Ƭ�ϼ�2,319,751,708�ֽڣ�������ͼƬ�ϼ�4,829,967,179�ֽڣ���ѹ����Ҫ����ռ䡣��ѡ�ļ�Ĭ�����ص� `offline_assets/`��Ҳ���� `--asset-root PATH` ָ��λ�á�

```bash
python wouaf.py setup --with-evaluation
python wouaf.py evaluate --version v3 --out outputs/independent_v3
python wouaf.py evaluate --version v1 --out outputs/independent_v1
python wouaf.py evaluate --version v2 --out outputs/independent_v2
```

[��������ָ��](docs/PUBLIC_REPRODUCTION.md)����������ѱ������ͼԤ������ͳ�ƣ��Լ�������¶�ȡͼƬ������ԭ��������ѵ�����ݺ�5000��latent�ǵ������о��ʲ���ʹ��ѵ���õĳ�Ʒ������̶�ͼƬ��������Ҫ���ǡ�`00_PRETRAINED_V3_DEMO.ipynb` ����ʷ����Drive�ָ���ڣ�������ͨ�û���װ���衣

### 7. ��Ŀ�ļ���ѵ��Ȩ��������ѵ��

| �ļ�/Ŀ¼ | ��; |
|---|---|
| `start_ui.cmd`��`wouaf.py` | �ճ���ҳ����������� |
| `models/` | ѵ���õ�V1/V2���Ȩ�أ�V3ʹ����������� |
| `configs/`��`ecc_wouaf/`��`ecc/` | �汾ѡ�񡢶�ȡͼƬ���ָ�ָ�ƺ͹̶��жϹ��� |
| `environment/portable/` | ���ػ�����װ�������������ļ� |
| `environment/locked/` | ��ʷ�ο���������ļ� |
| `training/` | ��ѵ������ģ��ʱʹ�õĿ�ѡ���� |
| `reports/`��`docs/` | ʵ�顢��������ƺ͸���˵�� |
| `outputs/` | ��������밲װ��¼ |

**Ȩ��**����ѵ��ѧ���Ĳ�����������`.pth`�ļ����Ŀ�ṩ��Ȩ��������Ϊѵ���õĲ�Ʒ��ȱʧʱ��װ���Զ����أ��������û�����ѵ����Ҳ��������ʹ�����ģ�͡�ԭ�ļ����������� [v3.0.0ģ��Release](https://github.com/Montos-Qu/ecc-protected-wouaf/releases/tag/v3.0.0)��

| �ļ� | SHA256У��ֵ |
|---|---|
| `models/v1_negative_aware_2400.pth` | `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb` |
| `models/v2_jpeg420_expert_extra800.pth` | `7a5f5b5b3149dd4ebd2837e3a36367fbb3cda4691935ca017c24fd403b48827d` |

����ѵ���Ƕ����Ŀ�ѡ���ܣ���[ѵ��˵��](docs/TRAINING.md)����ͨ���á���ҳ��Ԥ�������������ѵ�������ļ�ͨ��GitHub Releases�򹫿�Drive�ַ�����������ͨGit��ʷ��

### 8. ԭʼ WOUAF ������

��Applied Project����[kylemin��WOUAFԭ��Ŀ](https://github.com/kylemin/WOUAF)����ӦCVPR2024���������ǵ���չ������������żУ����ܾ����ԡ�V1/V2/V3���ա����ذ�װ����ҳ���档ԭ������ʵ�ֹ�ԭ���ߡ�ѧ��������ʹ��[����Citation](https://github.com/kylemin/WOUAF#citation)�����Ķ�[����˵��](docs/REFERENCES.md)��[������֪ͨ](THIRD_PARTY_NOTICES.md)���������ز��ı�ԭģ�͡�ͼƬ�������������������

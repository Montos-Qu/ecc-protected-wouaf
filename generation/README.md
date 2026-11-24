# Frozen baseline-3000 generator reproduction

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

This folder is for researchers who want to generate the original watermarked test images. You can skip it when using the app to read an existing image. The saved generator and latent files are optional research downloads, not files needed by normal setup. The details below explain how to replay those older experiments.

### Experiment details

Default pretrained attribution does not load the Stable Diffusion generator. The separate `baseline_generator` research asset includes SD2 UNet/text encoder/tokenizer/original PNDM scheduler, VAE, checkpoint3000, mapping network, original training source/commit/diff/environment records and the pristine VAE. Restore optional assets with:

```bash
python scripts/materialize_assets.py --asset-root "$ASSETS" --id baseline_generator independent_new_latents
```

`from_latents.py` restores custom VAE/mapping sources and renders with the original checkpoint and 32-bit codewords. It reads archived models, uses float32, disables TF32, preserves cudnn deterministic=False, latent scale .18215, mapping input bits+1e-8 and the original uint8 rounding. The independent cohort used batch4; this entry retains that batch size and exports PNGs with GT/latent/file hashes. Check the root research guide's limit4 command before explicitly requesting the full run.

`sample_latents.py` resamples from saved new prompts/seeds using local SD2 components, PNDM50, CFG7.5, per-unit seeded CPU initial noise and batch8. It does not automatically download new models or upgrade dependencies. Sampling all1024 units takes longer than restoring saved latents and is not part of the default demo.

For the exact verified JPEG bytes, restore `independent_encoded_jpeg_q90_444/420` directly. `encode_views.py` explicitly creates PNG/JPEG Q90 444/420 view manifests from PNG sources for historical regression or custom labeled data; default inference reads existing files without this reencoding.

`validation_latents` preserves the original5000 WOUAF latent tar.gz with its recorded directory structure, separately from the final1024 fresh latents. Original COCO natural photos are in `independent_natural_originals`, with cropped PNGs stored separately to preserve playback against future download/codec changes.

Original upstream full-training source and the3000-step checkpoint are research references under the restored asset's `code/` and `checkpoint-3000/`. The release verified portable extractor training and frozen generator restoration; it does not claim one-command exact replay of every upstream training branch. Optional generator/training/latent assets retain their separate access and licensing terms; they are not required for the pretrained product or public fixed-image evaluation.

---

<a name="chinese"></a>

## ����

���������������ɴ�ָ�Ƶ�ʵ��ͼƬ����ͨ�û�ֻ���ϴ�����ͼƬʶ��ʱ��������������Ŀ¼����������latent������ʱʹ�õ��м����ݣ����ڿ�ѡ�о����أ����ǰ�װ��Ʒ�ı����ļ������汣��������ʵ��ľ��������

### ʵ������

Ĭ�Ϲ�������Ҫ����SD�����������������ʲ���Ϊ `baseline_generator` �������棬����SD2��UNet/text encoder/tokenizer/ԭPNDMscheduler��VAE��checkpoint3000��mapping��ԭѵ���ű�/commit/diff/������¼��pristine VAE���ָ���

```bash
python scripts/materialize_assets.py --asset-root "$ASSETS" --id baseline_generator independent_new_latents
```

`from_latents.py`�ָ��Զ���VAE/mappingԴ�룬��ԭcheckpoint��32-bit������Ⱦ������ֻ���鵵ģ�ͣ�float32��TF32�رգ�cudnn deterministic=False��latent����.18215��mapping����bits+1e-8����ԭ��ʽ��������uint8��ԭ��������ÿ��4�ţ�����ڱ���ͬһbatch�����PNG����ͼGT/latent/file hash������README��limit4����˶ԣ�����ȷȥ��limit��ȫ����

`sample_latents.py`�ӱ������prompt/seed�嵥�ز�����ʹ�ñ���SD2�����PNDM50steps��CFG7.5��CPU����Ԫseed���ɳ�ʼ������batch8��Ĭ��û����������ģ�ͻ���������������1024��Ԫ������ʱ���Գ���ֱ�ӻָ��Ѵ�latent������Ĭ��demo�Ĳ��衣

���豣������֤ԭ��JPEG�ֽڣ�ֱ�ӻָ� `independent_encoded_jpeg_q90_444/420`��`encode_views.py`�ɴ�PNGԴ��ʽ����PNG/JPEGQ90 444/420�����嵥�����ھɻع�Դͼ���û��Լ��Ĵ���ǩ���ݣ�����Ĭ��inferֻ��ԭ�ļ���ͬ��

`validation_latents`����ԭ5000��WOUAF latent tar.gz��������Ŀ¼�ṹ��ѹ������������1024��latent�ֿ���ԭCOCO��Ȼͼ�ڶ����ʲ� `independent_natural_originals`���ü�PNG���棬����δ������/�����仯�����طš�

��������WOUAF��ͷѵ�������3000���ϵ�ɲ�ָ��ʲ� `code/`��`checkpoint-3000/`����������ԭѵ����Դ���о��ϵ㡣��������֤����ȡ�������̶��������ָ���Χ��û�н���������ȫѵ����֧����Ϊ��һ���ϸ��֡�

# Final V3: file-format-aware trained dual-extractor system

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

The final V3 keeps the better choice for each format: PNG/JPEG444 use V1, JPEG420 uses V2. Users only need to select V3; the app makes that choice automatically. The table below is an earlier regression test, not the later independent final test shown in the root README. Both records are kept, with their different datasets clearly identified.

### Experiment details

This is a system-level improvement: PNG/JPEG444 use V1, actual JPEG420 uses the trained V2 expert. Routing reads header sampling factors rather than filename, GT, user ID or test labels. BCH, confidence and presence policies remain fixed.

The extra1600-step single-model attempt and subsequent weight interpolation failed simultaneous retention and remain separate controls. This routed combination adds no training updates and does not claim one network improved every format.

Historical4096-source regression cohort,1024 positives per code/condition: this study reencoded views, checked actual headers and pixels, and executed12288 network forwards.

| Code | Condition | V1 correct | V3 correct | Change |
|---|---|---:|---:|---:|
| BCH(31,11)+P | PNG |930/1024 (90.82%)|930/1024 (90.82%)|0.00 pp|
| BCH(31,11)+P | JPEG444 Q90 |897/1024 (87.60%)|897/1024 (87.60%)|0.00 pp|
| BCH(31,11)+P | JPEG420 Q90 |755/1024 (73.73%)|787/1024 (76.86%)|+3.12 pp|
| BCH(31,16)+P | PNG |729/1024 (71.19%)|729/1024 (71.19%)|0.00 pp|
| BCH(31,16)+P | JPEG444 Q90 |597/1024 (58.30%)|597/1024 (58.30%)|0.00 pp|
| BCH(31,16)+P | JPEG420 Q90 |333/1024 (32.52%)|393/1024 (38.38%)|+5.86 pp|

PNG/444 per-image bits and acceptance decisions exactly match V1. False attributions including wrong positive identities and unwatermarked acceptance, summed across conditions: k11 V1 0 -> V3 0; k16 V1 1 -> V3 1. The combined fixed selection/calibration/regression check passed.

The router needs the actual encoded file. A JPEG decoded/saved as PNG no longer identifies its former420 sampling and uses V1. An RGB tensor alone cannot supply reliable sampling provenance. JPEG422/grayscale/other formats conservatively use V1 without the same effectiveness claim. Quantification covers512x512 Pillow Q90 444/420; other qualities/sizes require separate tests.

The development routing rule followed observed V1/V2 and failed single-model results. Selection/calibration and the4096-source cohort had been seen historically and are not a fresh independent test. PNG/444 retention follows the same V1 and pixels, not a single-model generalization result. Two fixed checkpoints are stored, one network runs per image. See the current root README for later fresh independent validation and default product use.

### Loading trained models

Current product entry, from the repository root:

```bash
python wouaf.py setup
python wouaf.py predict --image YOUR_ACTUAL_FILE.jpg --code 11 --output outputs/v3.json
```

The historical raw-delivery workflow extracted `V3_FORMAT_ROUTED_DELIVERY.zip` into `experiments/v3_format_router` in the earlier V1 portable repository, then used its `format_router_v3.py` through the locked launcher. That archived path is a historical reference, not required by the current product.

The router reads actual headers without a `--condition` argument, extra JPEG encoding or silent resizing. Output records version/branch, actual container/sampling, weights SHA256, logits/probabilities/bits and attribution. k16 is the65536-user control.

The old V1 `infer.py --condition jpeg_q90_420/444` actively reencoded at Q90; its default `png8_control` read actual files. The current V3 entry only reads encoded files. A description of that argument in the failed single-model raw delivery was incorrect; this explanation supersedes it.

The raw delivery retained both trained weights, inference code, per-image records, protocols, audits, demo and restore notebook. No retraining is needed. Full1600-step optimizer/RNG/checkpoints and failed candidates remain in the separate historical directory. Large original runtime/generator assets were not duplicated into that raw experiment; the current product obtains its required public runtime automatically. Original historical V1/V2 assets remain unchanged.

---

<a name="chinese"></a>

## ����

����V3����ʽѡ������ʵ������PNG/JPEG444��V1��JPEG420��V2���û�ֻҪѡV3��������Զ�ѡ���±��ǽ����һ�ֻع��飬���Ǹ�Ŀ¼README�к����Ķ������ղ��ԣ��������ݼ���ͬ����˶��������ֱ�˵����

### ʵ������

��ʷʵ���¼����ǰ��ƷĬ�ϰ汾���ֿ��README�е�����V3�����±����ý׶εķ����ͽ����

## V3�����ļ���ʽ����Ӧ����ѵ��˫��ȡ��

����Ϊϵͳ���Ż���PNG��JPEG444ʹ��V1��ʵ��JPEG420�ļ�ʹ����ѵ��V2ר�ҡ�·�ɶ�ȡ�ļ�ͷ�еĲ������ӣ������ļ�����GT���û�ID����Ա�ǩ��BCH�����Ŷ���presence�ż���ȫ�̶���

V3׷��1600���ĵ�ģ��ѵ�������Ȩ���ں�δͬʱ���㱣��Ҫ�󣬾�������������ǰ���û������ѵ�����£�Ҳ������ĳһ��ģ��ͬʱ������ȫ����ʽ��

����Ϊ��4096Դͼ�ع���У�ÿ��������ÿ��������1024�����������������±��롢���ʵ���ļ�ͷ���˶�ȫ�����ز�����ִ����12288��������ǰ��

| ���� | ���� | V1��ȷ�ָ� | V3��ȷ�ָ� | �仯 |
|---|---|---:|---:|---:|
| BCH(31,11)+P | PNG | 930/1024 (90.82%) | 930/1024 (90.82%) | +0.00���ٷֵ� |
| BCH(31,11)+P | JPEG444 Q90 | 897/1024 (87.60%) | 897/1024 (87.60%) | +0.00���ٷֵ� |
| BCH(31,11)+P | JPEG420 Q90 | 755/1024 (73.73%) | 787/1024 (76.86%) | +3.12���ٷֵ� |
| BCH(31,16)+P | PNG | 729/1024 (71.19%) | 729/1024 (71.19%) | +0.00���ٷֵ� |
| BCH(31,16)+P | JPEG444 Q90 | 597/1024 (58.30%) | 597/1024 (58.30%) | +0.00���ٷֵ� |
| BCH(31,16)+P | JPEG420 Q90 | 333/1024 (32.52%) | 393/1024 (38.38%) | +5.86���ٷֵ� |

PNG/444������Ԥ��bit�����/�ܾ����߾���V1��ȫһ�¡�����������������ϴ��û�����ˮӡͼ����գ��������ϼƣ���
- k11: V1 0 �� V3 0��
- k16: V1 1 �� V3 1��

�̶�selection/calibration���ع����ϼ�飺ͨ����

���ƣ��÷�����Ҫԭʼ�����ļ���JPEG�ȱ����벢����ΪPNG��PNG�ļ�ͷ�޷���ʾ��ȥ��420ѹ��������V1��ֻ��RGB����Ҳû�пɿ�����Ԫ���ݣ����ܳ�ŵ��ͬ���档JPEG422���Ҷ�JPEG��������ʽ������V1��δ��Ч����֤����ǰ���������Ӧ512��512��Pillow Q90��444/420����������������ߴ��������֤��

ʵ����ͣ�·�ɹ������ڹ۲�V1/V2��V3��ģ��ʧ�ܽ��������Ŀ���������selection/calibration��4096Դͼ��������ʷʵ��۲죬���ܳ�Ϊ�¶������ԡ�PNG/444��������ʹ����ͬV1ģ�������أ����ǵ�ģ��ѵ���ķ���֤������Ҫ�����̶�checkpoint������ͼִֻ������һ�����硣

### ������ѵ��ģ��

��ǰ��Ʒ������Ŀ��Ŀ¼ִ�������������ָ��ɽ�������

```bash
python wouaf.py setup
python wouaf.py predict --image YOUR_ACTUAL_FILE.jpg --code 11 --output outputs/v3.json
```

�����Ǿɽ���������ʷ�÷�����ͨ�û�����Ҫִ�У�

��V3_FORMAT_ROUTED_DELIVERY.zip��ѹ��ԭV1��Я�ֿ���Ŀ¼experiments/v3_format_router����V1 README�ָ���ͬ���ỷ�����ڲֿ��Ŀ¼���У�

```bash
python scripts/run_locked.py experiments/v3_format_router/format_router_v3.py --image YOUR_ACTUAL_FILE.jpg --code 11 --output outputs/v3.json
```

����ڴ�ʵ���ļ�ͷ�Զ�ѡ��֧������Ҫ--condition��Ҳ�������JPEG�ر����Ĭ���š����ͬʱ��¼model_version��selected_branch��ʵ��container/sampling��Ȩ��SHA��logits�����ʡ�bits�͹������ߡ�k=16��Ӧ65,536�û����ա�

ע��ԭV1��infer.py����--condition jpeg_q90_420/444��������������Q90�ر��룻ֱ�Ӷ�ȡ��ʵ�ļ�Ӧʹ����Ĭ��png8_control����V3�Զ����ֻ��ȡ�ļ���������ѹ�����ɵ�ģ��V3������LOAD_V3_DELIVERY.md�Ըò��������������Ա�˵��Ϊ׼��

V3_FORMAT_ROUTED_DELIVERY.zip����������ѵ��Ȩ�ء��Զ��������롢������GT/logits/����/bit/���ߡ�Э�顢��ơ���ʾ�ͻָ�notebook�������ٴ�ѵ��������1600��ѵ���ļ���/optimizer/RNG��ʧ�ܺ�ѡ�Ա������ϼ�V3Ŀ¼��������15GBԭ����/�������ʲ���������ԭV1 release/offline_assets�ָ���V1/V2��ԭĬ�����ñ��ֲ��䡣

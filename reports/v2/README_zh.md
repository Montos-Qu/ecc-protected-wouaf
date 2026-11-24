# V2 extension: full-codeword objective and JPEG420 recovery

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

What happened: further training helped JPEG420, but this candidate did not keep every required PNG/444 result. It was therefore not chosen as a replacement for V1 at that stage. Its trained JPEG420 specialist was later used as one component of the final V3. This folder records that earlier experiment; ordinary users can choose V3 in the app and skip these training details.

### Experiment details

Historical experiment record: this candidate failed the full predeclared retention requirements, so V1 remained the retained release at that stage. The current product defaults to final format-routed V3; V2 is its optional JPEG420 expert/control. No historical outcome is changed here.

Parent: `<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/negative_aware_multitask_20261029T020157Z`.
Parent model SHA256: `39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb`.

This independent experiment did not overwrite the previous GitHub preparation package. It uses the same dual-head architecture; `selected_multitask.pth` was the selected checkpoint. Plan/protocol locks record the fixed recipe/rules, training/selection histories record progress, and the last resumable checkpoint retains optimizer/RNG state.

The objective adds full BCH-codebook cross-entropy divided by32 to the dual-head loss, with V1 as a fit-only teacher. Actual view conditions cycle PNG/JPEG420/JPEG444/JPEG420/JPEG420. Group learning rates are1e-6/2.5e-6/5e-6/2e-5. Training adds2400 steps with100-step warmup and cosine scheduling; BN statistics and earlier layers are frozen.

Selection requires at least98% of V1's per-condition correct count for k11,95% for k16, and no bad acceptance for either code. Candidates are ordered by the fixed JPEG420 criteria, with the parent as step0 fallback. BCH/bit/presence thresholds never change; tests are not used for selection/tuning. Calibration roles had been observed historically, so this is not fresh independent data.

Regression tables are in `RESULTS_zh.html`/`regression_cells.csv`; per-image GT/logits/bits/presence/decisions are in `regression_v2_k*.npz`. Seeing no mistakes in this sample does not mean the system can never make a mistake. This experiment did not automatically replace the release; later independent content validation addresses generalization.

---

<a name="chinese"></a>

## ����

��������ʲô������ѵ���Ը���JPEG420������ѡû��ͬʱ��ס����Ҫ���PNG/444�������˵�ʱû�������滻V1����������V3ʹ��������ѵ���õ�JPEG420ר����������ﱣ����������ʵ���¼����ͨ�û��ڽ���ѡ��V3���ɣ�����Ҫ������������ѵ����

### ʵ������

��ʷʵ���¼����ǰ��ƷĬ�ϰ汾���ֿ��README�е�����V3�����±����ý׶εķ����ͽ����

## V2 ��չѵ������������Ŀ�꣫JPEG420�ָ�

δͨ��ȫ��Ԥ�ȹ̶��Ļع��ѡҪ��V1���Ǳ��������汾��V2�������Ϊʵ���¼��

���汾��<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/negative_aware_multitask_20261029T020157Z
��ģ��SHA256��39f57ba4864fefcd78f7ba7354d779df6adac209a0ec24b9eb774fe3560c1dfb

��Ŀ¼�Ƕ���ʵ��汾�����޸�֮ǰGitHub׼����������ģ��Ϊͬһ˫ͷ�ܹ���selected_multitask.pthΪѡ��Ȩ�أ�plan/protocol_lock��¼Ԥ�ȹ̶����䷽�͹���training_history/selection_history��¼���̣�last_resumable_checkpoint����optimizer/RNG��

Ŀ�꣺��ԭ˫ͷloss�����ϼ�������BCH�뱾�����أ�����32������ʹ�õ�ǰV1��Ϊfit-only��ʦ����ʵͼ��������PNG/JPEG420/JPEG444/JPEG420/JPEG420ѭ��������LRΪ1e-6/2.5e-6/5e-6/2e-5��2,400׷�Ӳ���100 warmup + cosine��BNͳ�ƺ�ǰ�㶳�ᡣ

ֻ��selectionͬʱ����k11>=98%��k16>=95%��ÿ������ȷ�������붼�޻�����ʱ���Ű�JPEG420��ȷ���ȹ̶�˳��ѡ�񣻸�ģ��Ҳ��Ϊstep0��ѡ��BCH/bit/presence�ż�ȫ�̶̹������Բ�����ѡ��/���ż���calibration��ɫ������ʷ�о�ʹ�ã���˴˴�Ҳ����ȫ�¶������ݡ�

��ǰ�ع����RESULTS_zh.html��regression_cells.csv��������GT��logits��bits��presence�;��߼�regression_v2_k*.npz���۲쵽�����������ʵ����ա�δ�Զ��滻����ģ�ͣ���Ҫ�¶��н�һ����֤������

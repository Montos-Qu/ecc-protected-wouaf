# V3 single-model attempt: three-view training and clean-view gradient retention

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

This was an unsuccessful attempt to make one model handle all three image formats better. It did not meet every requirement, so it is kept for comparison rather than used by the final app. Final V3 instead chooses between the already-trained V1 and V2. You do not need these failed candidates to run it.

### Experiment details

Historical failed candidate: it did not satisfy every fixed retention requirement, so V1 remained the retained release at that stage. This is not the final format-routed V3 product.

The model starts from V1 and trains only layer4 and the bit head; earlier layers, BN statistics and the presence head are frozen. Every step uses the same12 fit images as PNG/JPEG444/JPEG420 views. V1 teaches PNG/444 with logit and correct-bit-margin retention. V2 supplies JPEG420 teacher outputs only on fit images where its decoding is correct and GT BCE is lower. Gradient projection makes the first-order SGD update nonconflicting with both clean-view objectives. This does not guarantee unseen-image retention; development/regression checks remain necessary.

SGD uses no momentum or weight decay, LR.002,100-step warmup, cosine decay to10%, gradient norm clipping1,1600 steps, and selection every400 steps including step0. Both codes must have zero false attribution, PNG/444 correct counts no lower than V1, and420 no lower than V1. Ranking prioritizes k11 JPEG420, total correct count, then analogous k16 counts. Calibration/regression also require every-condition retention and strict primary-code420 improvement; rejection thresholds are fixed.

Old development and4096-source regression cohorts had been observed before and are not fresh independent tests. Fit/selection/calibration roles remain isolated. V1/V2 are read-only references; this separate experiment does not automatically replace the release. Checkpoints, GT, logits, probabilities, decisions, optimizer/RNG, training and selection records are retained.

---

<a name="chinese"></a>

## ����

����һ��û�гɹ��ĳ��ԣ�ϣ������ģ��ͬʱ�������ָ�ʽ�������û�дﵽȫ��Ҫ������ֻ���������գ�������������ҳӦ�á�����V3��Ϊ��ѵ���õ�V1/V2֮���Զ�ѡ�����г�Ʒ����Ҫ��Щʧ�ܺ�ѡ��

### ʵ������

��ʷʵ���¼����ǰ��ƷĬ�ϰ汾���ֿ��README�е�����V3�����±����ý׶εķ����ͽ����

## V3������ͼѵ����PNG/JPEG444�ݶȱ���

δͨ��ȫ��Ԥ�ȹ̶��Ļع��ѡҪ��V1���Ǳ��������汾��V3�������Ϊʵ���¼��

��V1��ʼ����ֻѵ��layer4��bit head��ǰ�㡢BNͳ�Ƽ�presence headȫ�����ᡣÿ��ͬһ12��fitͼ��ͬʱʹ��PNG��JPEG444��JPEG420������ͼ��V1��ΪPNG/444��ʦ����logit����ȷbit margin������V2ֻ��fitͼ�����ṩ��ȷ������GT BCE���͵�420��ʦ������ݶ�ͶӰʹSGD���������ݸɾ���ͼѵ��Ŀ��һ�ײ���ͻ���ⲻ��֤δ��ͼ���˻������Կ�����ع������顣

SGD��momentum����weight decay��ͳһLR0.002��100��warmup��cosine��10%���ݶȷ����ü�1����1600����ÿ400��ѡ�㣬���Ҳ��Ϊ��ѡ��ѡ��Ҫ�������޴��������PNG/444��ȷ����������V1��420������V1���ٰ�k11 JPEG420��ȷ��������ȷ����k16ͬ���������calibration��ع�ҲҪ��ÿ�������˻�������420�ϸ���ƣ��ܾ��ż���������

�ɿ�����4096Դͼ�ع���о�������ʷ�о��۲죬���ܳ�Ϊȫ�¶������ԡ�����fit/selection/calibration��ɫ���ָ��롣V1��V2����Ϊֻ���ο�����Ŀ¼�ǵ���V3�汾�����Զ��滻��������Ȩ�ء�GT��logits�����ʡ����ߡ��Ż���/RNG��ѵ����ѡ���¼��������

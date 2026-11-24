# V3 supplementary interpolation

[English](#english) | [����](#chinese)

<a name="english"></a>

## English

This follow-up mixed two sets of model parameters without more training. Some results improved and others fell, so it also failed the full target and was not used in the final app. The counts below show that trade-off; they are historical results, not the final V3 accuracy.

### Experiment details

Historical control: interpolation failed all strict goals; V1 remained the retained release at that stage. This is not the current final V3 system.

This supplement was proposed after observing a one-PNG loss on the training selection set and was not in the original training protocol. The interpolation grid was locked before candidate inference; only selection roles chose the magnitude. Calibration was checked once and the old4096-source cohort served as regression, not a fresh independent test. Seeing no mistakes in this sample does not mean the system can never make a mistake.

Source extra step:800. Layer4/bit-head mixing:(.5,.5). No new optimizer updates.

k11 V1 correct counts PNG/444/420:930/897/755; interpolated V3:931/895/756. False attribution0->0.

k16 V1 correct counts PNG/444/420:729/597/333; interpolated V3:723/598/335. False attribution1->1.

---

<a name="chinese"></a>

## ����

�����ڲ�׷��ѵ��������£�������ģ�Ͳ�����������ϡ����ֽ����ߣ������½�����δ�ﵽ����Ŀ�꣬���Ҳû����������Ӧ�á����汣���ļ�������˵������ȡ�ᣬ��������V3׼ȷ�ʡ�

### ʵ������

��ʷʵ���¼����ǰ��ƷĬ�ϰ汾���ֿ��README�е�����V3�����±����ý׶εķ����ͽ����

## V3����׶Σ�С��Ȩ���ں�

δ�ﵽȫ���ϸ�Ŀ�꣬����ʵ���¼��V1�����汣����

������׶����ڿ���ѵ��ѡ�񼯵�PNGһ����ʧ�������������ԭʼѵ��Э���һ���֡��ں��������κ��ںϺ�ѡ����ǰ������ֻ��selection��ɫѡ���ȡ�calibrationֻ����һ�Σ���4096Դͼֻ���ع飬�������¶������ԡ���������û�й۲쵽���󣬲������Ժ���Զ���������

Դ���ⲽ����800��layer4/bit-head�ںϱ�����(0.5, 0.5)���������Ż������¡�

k11 V1: {'png8_control': 930, 'jpeg_q90_444': 897, 'jpeg_q90_420': 755}��V3: {'png8_control': 931, 'jpeg_q90_444': 895, 'jpeg_q90_420': 756}���������0��0��

k16 V1: {'png8_control': 729, 'jpeg_q90_444': 597, 'jpeg_q90_420': 333}��V3: {'png8_control': 723, 'jpeg_q90_444': 598, 'jpeg_q90_420': 335}���������1��1��

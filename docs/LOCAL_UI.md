[English](#english) | [中文](#chinese)

<a name="english"></a>

# Local web interface

Desktop typography is enlarged, with a wider workspace and readable controls/results on2560x1440 displays. The layout also adapts to narrower windows.

On Windows, double-click `start_ui.cmd`. It prepares the portable environment when needed, starts a hidden local backend, and opens the default browser. No preinstalled Python is required. Existing installed runtimes and original trained weights are reused. First setup needs internet for official runtime binaries and any missing product assets; normal image inference is local.

1. Upload/drop a native512x512 PNG/JPEG, or choose a built-in sample. Expand the size explanation below the preview, or read [image size limits](IMAGE_SIZE_LIMITS.md). Automatic fitting is removed in3.0.5.
2. Keep V3 (default), or choose V1/V2. Optional user capacity:k11 default2048, k16 control65536.
3. Select **Identify fingerprint**. The first request may take longer to load weights.
4. Read **ACCEPTED** with its user ID, or **REJECTED** with the primary failed rule. No rejected candidate is presented as an accepted identity.
5. Download the full JSON for logits, probabilities,32 predicted bits, hashes and runtime identity. A local copy is saved under `outputs/ui/`.

**Stop app** shuts down the backend and releases its models. Closing a tab alone does not stop it. Double-clicking the launcher again reopens a healthy existing instance. The service binds only to127.0.0.1, normally port8765; an occupied port uses an available local port. The actual address is printed and recorded in `outputs/ui_server.json`. Logs are in `outputs/ui_stdout.log` and `outputs/ui_stderr.log`.

CLI on an installed Windows/Linux portable runtime:

```bash
python wouaf.py ui
python wouaf.py ui --device cpu --port 8766
python wouaf.py ui --no-browser
```

No extra web framework, CDN, external font, authentication service or package changes are required. The UI calls the original Predictor on original uploaded bytes. V3 routing follows actual encoded JPEG sampling, even for a renamed file. There is no silent resizing/recompression, training, threshold tuning, or modification of the scientific inference core. V1/V2/V3 share the same two immutable component models in memory. Maximum upload size:10MiB. Input images remain in memory; only the result JSON is saved. Model-generation/training controls are not part of this inference UI.

REJECT is an inference outcome, not an application error or proof that an image is unwatermarked. A positive JPEG may be rejected if correction fails. An ordinary photograph is a negative test, not a suitable positive attribution demonstration. The original512x512/Pillow Q90 PNG/JPEG444/JPEG420 validation scope remains unchanged; other JPEG sampling gets a visible scope notice. The UI does not claim new robustness or population error guarantees. Portable mode remains separate from strict historical A100 reference mode.

<a name="chinese"></a>

# 本地网页界面

已放大桌面正文、按钮、控件、结果与说明文字，并加宽工作区，适配2560x1440显示；窄窗口仍自动调整布局。

Windows双击 `start_ui.cmd`。需要时自动配置普通模式环境，后台启动本地服务，并打开默认浏览器，不必预装Python。复用已安装环境及原训练权重。首次配置需要联网获取官方运行包及缺失的项目资产，日常图片识别在本机完成。

1. 上传/拖入原生512x512 PNG/JPEG，或选择内置示例。预览下方可展开原因，另见[尺寸说明](IMAGE_SIZE_LIMITS.md)。3.0.5已取消自动缩放。
2. 默认V3，也可选择V1/V2。默认k11容量2048，k16为65536用户容量对照。
3. 点击“开始识别指纹”。首次请求加载模型可能稍慢。
4. 接受时显示用户ID；拒绝时解释最先未通过的规则，不将内部候选当成接受身份。
5. 下载完整JSON，其中保留logits、概率、32位指纹、hash和运行环境；本地副本保存在 `outputs/ui/`。

“关闭服务”停止后台并释放模型，仅关闭标签页不会退出服务。再次双击启动器会重新打开同一目录下的正常实例。服务只监听127.0.0.1，通常端口8765；被占用时选择空闲本地端口。实际地址记录在 `outputs/ui_server.json`；日志为 `outputs/ui_stdout.log` 和 `outputs/ui_stderr.log`。已安装普通模式也可使用以上CLI命令。

界面不增加网页框架、外部字体/CDN、身份服务或依赖变更。通过原Predictor读取上传的原始编码字节，改扩展名不影响实际JPEG格式路由；不缩放、不重新压缩、不训练、不调门槛、不改科学推理核心。各版本在内存共享原两份组件。上传上限10MiB，输入图片只留内存，仅保存结果JSON。此界面不包含生成器或训练控制。

拒绝是识别结果，不是程序崩溃，也不能证明图片无水印；带指纹JPEG可能因纠错失败被拒绝。普通照片适合负样本测试，不能演示已嵌入用户的正确恢复。原512x512/Pillow Q90 PNG/JPEG444/JPEG420验证范围不变，其他JPEG采样会显示范围提示。新增界面不等于新增鲁棒性或总体错误率保证，普通模式与历史A100严格复现仍然分开。

"use strict";
const copy = {
  en: {
    local: "Local workspace", stop: "Stop app", eyebrow: "ECC-PROTECTED WOUAF / PRETRAINED V3", title: "A fingerprint. A clearer answer.", subtitle: "Read a WOUAF user fingerprint from an image, with correction and careful rejection built in.", runtimeNote: "Runs on your computer", inputTitle: "Choose an image", drop: "Drop your image here", browse: "or click to browse files", requirements: "Native 512 × 512 · PNG or JPEG · Up to 10 MB", replace: "Click to replace image", trySample: "Try a sample", samplePng: "Watermarked PNG", sampleJpeg: "JPEG420", sampleNegative: "Unmarked PNG", modelTitle: "Choose a model", recommended: "Final", v3: "Format-aware routing", v1: "Balanced extractor", v2: "JPEG420 expert", capacity: "User capacity", k11: "2,048 users · BCH(31,11)+P · Default", k16: "65,536 users · BCH(31,16)+P · Control", analyze: "Identify fingerprint", privacy: "Images stay on this computer. No cloud upload. No training.", resultTitle: "Attribution result", emptyTitle: "Your result starts here", emptyText: "Choose an image and a model, then identify its fingerprint.", emptyNote: "Trained weights ready. No retraining needed.", working: "Reading the fingerprint…", workingText: "The first image may take longer while trained weights load.", usedModel: "Selected model", branch: "Extractor used", format: "Actual image format", ecc: "Correction scheme", bitsTitle: "Extracted fingerprint", outsideScope: "This JPEG sampling mode is outside the validated 4:4:4 / 4:2:0 scope.", download: "Download result JSON", technical: "View complete result", readTitle: "A careful decision", readText: "REJECT means no identity is accepted. A compressed watermarked image can also be rejected when recovery is unreliable.", footer: "A100-trained components. Local inference. Original weights and thresholds.", accepted: "ACCEPTED", rejected: "REJECTED", user: "User", noIdentity: "No identity accepted", acceptExplanation: "The fingerprint passed correction, confidence and watermark-presence checks.", reasonEcc: "The extracted bits could not be corrected within the BCH decoding radius. No user is attributed.", reasonConfidence: "The fingerprint did not meet the fixed confidence threshold. No user is attributed.", reasonPresence: "The watermark-presence check did not pass. No user is attributed.", reasonOther: "The image did not pass the attribution rules. No user is attributed.", clean: "V1 · Balanced", expert: "V2 · JPEG420 expert", stopped: "Local app stopped", stoppedText: "You can close this tab. Double-click start_ui.cmd to open the app again.", stopping: "Stopping…", stale: "Settings changed. Identify the fingerprint again to update the result.", cpu: "CPU · Ready", cuda: "NVIDIA CUDA · Ready", error: "Something went wrong. Please try again.", disconnected: "Cannot reach the local app. Double-click start_ui.cmd to start it again.", wrong_dimensions: "Please choose a native 512 × 512 image. The app does not resize images automatically.", unsupported_format: "Please choose a PNG or JPEG image.", invalid_image: "This file could not be read as a valid image. Please choose another PNG or JPEG.", image_too_large: "Please choose a nonempty image no larger than 10 MB.", session_expired: "This session has expired. Refresh the page and try again.", invalid_configuration: "Choose V3, V1 or V2 and one of the available user capacities.", inference_failed: "Recognition could not finish. Check outputs/ui_stderr.log, or restart the local app.", sample_missing: "This sample is missing. Run setup_windows.cmd to restore the supplied samples.", invalid_upload: "The image upload was not valid. Please choose the file again.", upload_timeout: "The upload timed out. Please try again.", local_access_only: "Open this app through its local 127.0.0.1 address."
  },
  zh: {
    local: "本地工作空间", stop: "关闭服务", eyebrow: "ECC-PROTECTED WOUAF / 已训练 V3", title: "从图片指纹，找到用户归属。", subtitle: "提取 WOUAF 用户指纹，通过纠错与拒绝规则，给出可靠范围内的识别结果。", runtimeNote: "全部在你的电脑运行", inputTitle: "选择图片", drop: "将图片拖到这里", browse: "或点击浏览本机文件", requirements: "原生 512 × 512 · PNG 或 JPEG · 最大 10 MB", replace: "点击更换图片", trySample: "试用示例", samplePng: "带指纹 PNG", sampleJpeg: "JPEG420", sampleNegative: "无指纹 PNG", modelTitle: "选择模型", recommended: "最终版", v3: "按实际格式自动选择", v1: "平衡提取器", v2: "JPEG420 专家", capacity: "用户容量", k11: "2,048 用户 · BCH(31,11)+P · 默认", k16: "65,536 用户 · BCH(31,16)+P · 对照", analyze: "开始识别指纹", privacy: "图片只在本机处理，不上传云端，不进行训练。", resultTitle: "用户归属结果", emptyTitle: "识别结果将在这里显示", emptyText: "选择一张图片和模型，然后开始识别用户指纹。", emptyNote: "直接使用已训练权重，无须重新训练。", working: "正在读取图片指纹…", workingText: "首次识别需要加载训练好的权重，可能稍慢一些。", usedModel: "所选模型", branch: "实际使用提取器", format: "实际图片格式", ecc: "纠错方案", bitsTitle: "提取的指纹", outsideScope: "此 JPEG 采样格式不在已验证的 4:4:4 / 4:2:0 范围内。", download: "下载完整 JSON 结果", technical: "查看完整结果", readTitle: "谨慎给出归属", readText: "拒绝意味着没有接受任何用户身份。带指纹图片压缩后，也可能因无法可靠恢复而被拒绝。", footer: "组件基于 A100 训练 · 本机推理 · 保留原权重与门槛", accepted: "接受归属", rejected: "拒绝归属", user: "用户", noIdentity: "未接受任何用户身份", acceptExplanation: "指纹已通过纠错、置信度和水印存在性检查。", reasonEcc: "提取的指纹位超出 BCH 可完成解码的范围，无法可靠纠错，因此不归属给任何用户。", reasonConfidence: "指纹未达到固定的置信度门槛，因此不归属给任何用户。", reasonPresence: "未通过水印存在性检查，因此不归属给任何用户。", reasonOther: "图片未通过归属规则，因此不接受任何用户身份。", clean: "V1 · 平衡提取器", expert: "V2 · JPEG420 专家", stopped: "本地服务已关闭", stoppedText: "可以关闭此页面。再次双击 start_ui.cmd 即可重新打开。", stopping: "正在关闭…", stale: "设置已更改，请重新识别以更新结果。", cpu: "CPU · 就绪", cuda: "NVIDIA CUDA · 就绪", error: "发生错误，请重试。", disconnected: "无法连接本地服务，请双击 start_ui.cmd 重新启动。", wrong_dimensions: "请选择原生 512 × 512 图片。程序不会自动缩放图片。", unsupported_format: "请选择 PNG 或 JPEG 图片。", invalid_image: "无法读取有效图片，请选择其他 PNG 或 JPEG 文件。", image_too_large: "请选择非空且不超过 10 MB 的图片。", session_expired: "当前会话已失效，请刷新页面后重试。", invalid_configuration: "请选择 V3、V1 或 V2，以及提供的用户容量。", inference_failed: "识别未能完成。可检查 outputs/ui_stderr.log，或重新启动本地服务。", sample_missing: "示例图片缺失，请运行 setup_windows.cmd 恢复随附样本。", invalid_upload: "图片上传无效，请重新选择文件。", upload_timeout: "上传超时，请重试。", local_access_only: "请通过本机 127.0.0.1 地址打开。"
  }
};
Object.assign(copy.en, {
  nativeRequirements: "Original mode: 512 × 512 · PNG/JPEG · Up to 10 MB", fitRequirements: "Auto fit enabled · PNG/JPEG · Up to 10 MB",
  resizeLabel: "Image sizing", resizeNative: "Original 512 × 512 · Default", resizeFit: "Fit to 512 + padding · Experimental",
  resizeHint: "Optional fitting preserves aspect ratio. Original 512 images remain untouched.",
  resizeWarning: "Experimental preprocessing: resizing can damage fingerprints. Accepting a file size does not establish reliable recovery.",
  resizeUnchanged: "Auto fit selected; this 512 × 512 input was not modified.", resizePreview: "Processed preview", resizeInput: "Input", resizeContent: "Fitted content", resizeModel: "Model input",
  too_many_pixels: "Choose an image up to 16,777,216 pixels total, with neither side above 8,192 pixels."
});
Object.assign(copy.zh, {
  nativeRequirements: "原始模式：512 × 512 · PNG/JPEG · 最大10 MB", fitRequirements: "已启用自动缩放 · PNG/JPEG · 最大10 MB",
  resizeLabel: "图片尺寸处理", resizeNative: "原始 512 × 512 · 默认", resizeFit: "等比缩放至 512 + 补边 · 实验",
  resizeHint: "仅在选择自动缩放时等比处理，必要时补边；原始512图片保持不变。",
  resizeWarning: "实验性预处理：缩放可能损伤指纹。能够接收此尺寸，不代表可以可靠恢复身份。",
  resizeUnchanged: "已选择自动缩放；该图片本身为512 × 512，没有修改像素。", resizePreview: "处理后预览", resizeInput: "原图", resizeContent: "缩放后内容", resizeModel: "模型输入",
  too_many_pixels: "请选择总像素不超过16,777,216、且任一边不超过8,192像素的图片。"
});
const $ = id => document.getElementById(id);
let language = localStorage.getItem("wouaf-language") || (navigator.language.startsWith("zh") ? "zh" : "en");
let selectedFile = null;
let previewUrl = null;
let lastResponse = null;
let busy = false;
let stopped = false;
let lastError = null;
let device = "cpu";
const token = document.querySelector('meta[name="wouaf-token"]').content;
const t = key => copy[language][key] || copy[language].error;

function renderLanguage() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach(node => { node.textContent = t(node.dataset.i18n); });
  $("language").textContent = language === "zh" ? "English" : "中文";
  $("device").textContent = stopped ? t("stopped") : t(device);
  $("dropzone").setAttribute("aria-label", t("inputTitle"));
  document.querySelector('[data-i18n="requirements"]').textContent = t($("resize").value === "fit" ? "fitRequirements" : "nativeRequirements");
  if (lastResponse) renderResult();
  if (lastError) $("error").textContent = t(lastError);
}
function showError(key) {
  lastError = key;
  $("error").textContent = t(key);
  $("error").hidden = false;
}
function clearResult() {
  lastResponse = null;
  lastError = null;
  $("result").hidden = true;
  $("error").hidden = true;
  $("empty").hidden = false;
  $("resize-result-note").hidden = true;
  $("processed-info").hidden = true;
  if (previewUrl) $("preview").src = previewUrl;
  $("model-badge").textContent = document.querySelector('input[name="version"]:checked').value.toUpperCase();
}
function chooseFile(file) {
  if (busy || stopped || !file) return;
  clearResult();
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("preview").hidden = false;
  $("preview").onerror = () => { $("preview").hidden = true; };
  $("upload-placeholder").hidden = true;
  $("replace").hidden = false;
  $("file-info").hidden = false;
  $("filename").textContent = file.name;
  $("filesize").textContent = (file.size / 1024).toFixed(0) + " KB";
  $("predict").disabled = !file.size || file.size > 10 * 1024 * 1024;
  if ($("predict").disabled) showError("image_too_large");
}
function renderResult() {
  const {result: r, elapsed_ms: elapsed, download_url: download} = lastResponse;
  const accepted = r.decision === "ACCEPT";
  $("decision-card").classList.toggle("reject", !accepted);
  $("decision").textContent = t(accepted ? "accepted" : "rejected");
  $("identity").textContent = accepted ? t("user") + " " + r.user_id : t("noIdentity");
  const reason = {outside_BCH_decoding_radius: "reasonEcc", bit_confidence_below_threshold: "reasonConfidence", watermark_presence_below_threshold: "reasonPresence"}[r.reject_reason] || "reasonOther";
  $("explanation").textContent = t(accepted ? "acceptExplanation" : reason);
  $("elapsed").textContent = (elapsed / 1000).toFixed(2) + " s";
  $("used-model").textContent = r.requested_version.toUpperCase();
  $("model-badge").textContent = r.requested_version.toUpperCase();
  $("branch").textContent = t(r.selected_branch === "jpeg420_expert" ? "expert" : "clean");
  const sampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}[r.jpeg_sampling];
  $("format").textContent = r.container + (r.container === "JPEG" ? " · " + (sampling || "other") : "");
  $("ecc").textContent = r.code;
  $("bits").replaceChildren(...r.pred_bits.map((bit, index) => {
    const node = document.createElement("span");
    node.className = "bit" + (bit ? " on" : ""); node.textContent = bit; node.title = "Bit " + index; return node;
  }));
  $("scope-note").hidden = r.validated_container_sampling;
  $("download").href = download;
  $("raw").textContent = JSON.stringify(r, null, 2);
  const sizing = r.preprocessing;
  $("resize-result-note").hidden = !sizing;
  if (sizing) $("resize-result-note").textContent = t(sizing.applied ? "resizeWarning" : "resizeUnchanged");
  $("processed-info").hidden = !sizing || !sizing.applied;
  if (sizing && sizing.applied) {
    $("processed-info").textContent = t("resizePreview") + " · " + t("resizeInput") + " " + sizing.source_size.join(" × ") + " → " + t("resizeContent") + " " + sizing.fitted_size.join(" × ") + " → " + t("resizeModel") + " 512 × 512";
    $("preview").src = lastResponse.processed_preview;
  }
}
async function predict() {
  if (!selectedFile || busy || stopped) return;
  clearResult();
  busy = true;
  document.querySelectorAll("button,input,select").forEach(node => { node.disabled = true; });
  $("predict").classList.add("processing"); $("empty").hidden = true; $("busy").hidden = false;
  const query = new URLSearchParams({version: document.querySelector('input[name="version"]:checked').value, code: $("code").value, resize: $("resize").value});
  try {
    const response = await fetch("/api/predict?" + query, {method: "POST", headers: {"Content-Type": "application/octet-stream", "X-WOUAF-Token": token, "X-Filename": encodeURIComponent(selectedFile.name)}, body: selectedFile});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "error");
    lastResponse = data;
    renderResult(); $("result").hidden = false;
  } catch (error) {
    showError(error instanceof TypeError ? "disconnected" : error.message);
    $("empty").hidden = false;
  } finally {
    busy = false; $("busy").hidden = true; $("predict").classList.remove("processing");
    document.querySelectorAll("button,input,select").forEach(node => { node.disabled = false; });
  }
}
$("file").addEventListener("change", event => chooseFile(event.target.files[0]));
$("dropzone").addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!busy && !stopped) $("file").click(); } });
$("dropzone").addEventListener("dragover", event => { event.preventDefault(); $("dropzone").classList.add("dragging"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
$("dropzone").addEventListener("drop", event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); chooseFile(event.dataTransfer.files[0]); });
$("predict").addEventListener("click", predict);
$("language").addEventListener("click", () => { language = language === "en" ? "zh" : "en"; localStorage.setItem("wouaf-language", language); renderLanguage(); });
document.querySelectorAll('input[name="version"],#code,#resize').forEach(node => node.addEventListener("change", () => { clearResult(); renderLanguage(); }));
document.querySelectorAll("[data-sample]").forEach(button => button.addEventListener("click", async () => {
  if (busy || stopped) return;
  try {
    const response = await fetch("/api/sample?id=" + button.dataset.sample);
    if (!response.ok) { const data = await response.json(); throw new Error(data.error); }
    const blob = await response.blob();
    chooseFile(new File([blob], button.dataset.sample + (blob.type === "image/jpeg" ? ".jpg" : ".png"), {type: blob.type}));
  } catch (error) { showError(error instanceof TypeError ? "disconnected" : error.message); }
}));
$("stop").addEventListener("click", async () => {
  if (busy || stopped) return;
  $("stop").textContent = t("stopping");
  try {
    const response = await fetch("/api/shutdown", {method: "POST", headers: {"X-WOUAF-Token": token}});
    if (!response.ok) throw new Error("session_expired");
    stopped = true; clearResult(); $("stop").textContent = t("stopped");
    $("empty").querySelector("h3").textContent = t("stopped");
    $("empty").querySelector("p").textContent = t("stoppedText");
    $("device").textContent = t("stopped");
    document.querySelectorAll("button,input,select").forEach(node => { node.disabled = true; });
  } catch (error) { showError(error instanceof TypeError ? "disconnected" : error.message); $("stop").textContent = t("stop"); }
});
renderLanguage();
fetch("/api/health").then(response => response.json()).then(health => { device = health.device; renderLanguage(); }).catch(() => showError("disconnected"));

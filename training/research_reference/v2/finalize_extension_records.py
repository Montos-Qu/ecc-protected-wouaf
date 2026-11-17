"""Save the human review, version index and next-session inference entry; no training."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--folder-url',required=True);a=p.parse_args();root=a.root
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
plan=json.loads((root/'plan.json').read_text());decision=json.loads((root/'decision.json').read_text())
audit=json.loads((root/'independent_audit.json').read_text());portable=json.loads((root/'portable_inference_verification.json').read_text())
assert audit['complete'] and portable['complete']
cal=json.loads((root/'calibration_check.json').read_text());summary=json.loads((root/'regression_summary.json').read_text())
failures=[]
for k,v in cal.items():
 target=.98 if k=='11' else .95
 for condition,retention in v['retention'].items():
  if retention<target:failures.append(f"k{k} {condition}: {v['parent']['correct_by_condition'][condition]} → {v['metrics']['correct_by_condition'][condition]}, 保留率{100*retention:.2f}%，低于{100*target:.0f}%要求")
 if v['metrics']['bad']>0:failures.append(f"k{k} calibration错误归属{v['metrics']['bad']}")
reason='；'.join(failures) if failures else '所有开发保留检查通过；仍需按完整回归和独立新队列判断。'
lines=['# V2扩展训练：最终复核','',f"版本：{root.name}",f"完成追加训练：2400步；选择的额外步数：{decision['selected_additional_step']}。",'',
 '当前状态：独立实验候选，未替换V1发布模型。','',
 '| 编码 | 条件 | V1正确/1024 | V2正确/1024 |','|---|---|---:|---:|']
for entry in summary:
 for condition in ['png8_control','jpeg_q90_444','jpeg_q90_420']:
  lines.append(f"| k{entry['k']} | {condition} | {entry['parent_v1']['correct_by_condition'][condition]} | {entry['extension_v2']['correct_by_condition'][condition]} |")
lines+=['','错误归属总数（正样本认错用户与无水印图被接受，合计三种条件）：']
for entry in summary:lines.append(f"- k{entry['k']}: V1 {entry['parent_v1']['bad']} → V2 {entry['extension_v2']['bad']}。")
lines+=['','未通过的预设开发检查：'+reason,'',
 'JPEG420恢复有所提升，PNG与JPEG444有所下降，因此不能把V2称为全面更好的正式版本。原4096源图是已观察过的回归队列，不是新独立测试；零观察错误也不代表真实错误概率为零。','',
 '审计通过：2400个fit批次、231个冻结参数/BN状态项、36864个双版本决策、36个结果单元格；V1发布ZIP与所有冻结输入未变。V2候选解压到新V1便携副本后，以独立配置加载通过，默认V1文件逐一hash不变。','',
 'CANDIDATE_BUNDLE.zip包含实际已训练的V2选中权重与独立inference_config.json。按LOAD_V2_IN_V1.md放到V1仓库的experiments/v2目录，通过显式--config调用。默认不带--config仍使用V1。','',
 '本目录保留所有六个额外检查点、optimizer/RNG、逐样本GT/logits/probabilities/决策、训练和选择历史。研究plan包含原Drive路径，属于私有实验记录；V1预训练发布包仍是之前整理的GitHub准备版本。','',
 '下一次恢复：打开RESTORE_V2_INFERENCE.ipynb，挂载同一Drive，恢复同一冻结环境后复核；该入口不训练。完整训练已完成，不要直接重复启动wouaf_extension_v2.py。','',a.folder_url]
(root/'FINAL_REVIEW_zh.md').write_text('\n'.join(lines)+'\n')
page=(root/'RESULTS_zh.html').read_text().replace('</html>','<p><strong>未通过检查：</strong>'+reason+'</p><p>独立审计与换目录推理验证通过。V2已保存为实验版本，V1保持原发布状态。</p></html>')
(root/'FINAL_REVIEW_zh.html').write_text(page)
def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def cell(text):return {'cell_type':'code','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]}
release=Path(plan['parent_release_root'])
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
 md('# 恢复V2实验候选推理\n本版本尚未通过全部保留要求，V1发布版保持。以下只恢复软件与加载已训练权重，不进行训练。参考A100-SXM4-80GB/driver580.82.07；不同硬件会被严格检查拦住。'),
 cell('from google.colab import drive\ndrive.mount("/content/drive")\n'),
 cell('from pathlib import Path\nimport json,zipfile,subprocess,sys\nRELEASE=Path('+repr(str(release))+')\nV2=Path('+repr(str(root))+')\nWORK=Path("/content/wouaf_v2_inference")\nWORK.mkdir(exist_ok=True)\nBASE=WORK/"ecc-protected-wouaf"\nif not BASE.exists():\n    with zipfile.ZipFile(RELEASE/"ecc-protected-wouaf-pretrained.zip") as z:z.extractall(WORK)\nVERSION=BASE/"experiments/v2"\nVERSION.mkdir(parents=True,exist_ok=True)\nwith zipfile.ZipFile(V2/"CANDIDATE_BUNDLE.zip") as z:z.extractall(VERSION)\n'),
 cell('subprocess.run([sys.executable,str(BASE/"scripts/materialize_assets.py"),"--asset-root",str(RELEASE/"offline_assets"),"--id","frozen_environment"],check=True)\nsubprocess.run([sys.executable,str(BASE/"scripts/restore_environment.py")],check=True)\nsubprocess.run([sys.executable,str(BASE/"scripts/run_locked.py"),"--check"],check=True)\n'),
 cell('example=next(r for r in json.loads((BASE/"demo/manifest.json").read_text()) if r["group"]=="positive_k11")\noutput=BASE/"outputs/v2_prediction.json"\noutput.parent.mkdir(exist_ok=True)\nsubprocess.run([sys.executable,str(BASE/"scripts/run_locked.py"),str(BASE/"infer.py"),"--config",str(VERSION/"inference_config.json"),"--image",str(BASE/example["image"]),"--condition","jpeg_q90_420","--code","11","--output",str(output)],check=True)\nprint(output.read_text())\n'),
 md('完整结果与采用限制见FINAL_REVIEW_zh.md；不修改原V1权重、默认配置或全局工作方案。')]}
save(root/'RESTORE_V2_INFERENCE.ipynb',notebook)
index=root.parent/'VERSION_INDEX.json';entries=json.loads(index.read_text())
for entry in entries:
 if entry['root']==str(root):entry.update({'drive_folder_url':a.folder_url,'portable_inference_verified':True,
  'release_status':'experimental_not_promoted' if not decision['regression_promising'] else 'experimental_regression_candidate',
  'review':'FINAL_REVIEW_zh.md','restore_entry':'RESTORE_V2_INFERENCE.ipynb'})
save(index,entries)
pointer=root.parent.parent/'LATEST_EXTENSION_TRAINING.json';value=json.loads(pointer.read_text());value.update({'drive_folder_url':a.folder_url,
 'independent_audit_passed':True,'portable_inference_verified':True,'release_status':entries[-1]['release_status']});save(pointer,value)
proof=['FINAL_REVIEW_zh.md','FINAL_REVIEW_zh.html','RESTORE_V2_INFERENCE.ipynb','portable_inference_verification.json','portable_v2_example.json','CANDIDATE_BUNDLE.zip']
(root/'DELIVERY_SHA256SUMS.txt').write_text('\n'.join(f'{sha(root/n)}  {n}' for n in proof)+'\n')
save(root/'delivery_complete.json',{'complete':True,'version':root.name,'trained_extra_steps':2400,'selected_extra_step':decision['selected_additional_step'],
 'release_status':entries[-1]['release_status'],'parent_release_unchanged':True,'independent_audit_passed':True,'portable_inference_passed':True,
 'drive_folder_url':a.folder_url,'gate_failure_explanation':reason,'new_independent_cohort_tested':False})
print('V2_RECORDS_FINALIZED',json.dumps(json.loads((root/'delivery_complete.json').read_text()),ensure_ascii=False),flush=True)

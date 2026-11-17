"""Finish verification and save one explicit V3 delivery; never updates V1 or V2."""
import argparse,datetime,hashlib,html,json,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--launcher',required=True);p.add_argument('--folder-url',required=True);a=p.parse_args();root=a.root
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text())
def run(script,target,log):
 with log.open('w') as f:r=subprocess.run([sys.executable,a.launcher,str(script),'--root',str(target)],stdout=f,stderr=subprocess.STDOUT,timeout=600)
 assert r.returncode==0,log.read_text()[-6000:]
def wait_for(path,allowed,log=None):
 start=time.monotonic()
 while True:
  state=read(path)
  if state['stage'] in allowed:return state
  if state['stage']=='failed':raise RuntimeError(str(state))
  if log and log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError(log.read_text()[-6000:])
  assert time.monotonic()-start<2400,'Bounded verification wait expired.'
  time.sleep(5)
try:
 plan=read(root/'plan.json');local=Path(plan['local']);parent=Path(plan['parent_root']);v2=Path(plan['v2_root'])
 wait_for(root/'status.json',{'complete'})
 save(root/'delivery_status.json',{'stage':'independent_audit'})
 # Preserve the first failed checker and its output; the science protocol is unchanged.
 if not (root/'audit_status_initial_failure.json').exists():
  shutil.copy2(root/'audit_status.json',root/'audit_status_initial_failure.json')
  shutil.copy2(local/'independent_audit.log',root/'independent_audit_initial_failure.log')
 correction={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'original_script_sha256':sha(root/'audit_extension_v3.py'),'corrected_script_sha256':sha(root/'audit_extension_v3_corrected.py'),
  'reason':'PyTorch 1.12.1 stateless SGD serializes one momentum_buffer: None entry per parameter. The initial checker incorrectly required an empty outer state mapping.',
  'corrected_assertion':'Uniform final LR 0.0002, momentum=weight_decay=0; exactly one state per optimized parameter, each exactly {momentum_buffer: None}.',
  'training_selection_gates_metrics_unchanged':True}
 save(root/'AUDIT_CORRECTION.json',correction)
 run(root/'audit_extension_v3_corrected.py',root,local/'independent_audit_corrected.log')
 shutil.copy2(local/'independent_audit_corrected.log',root/'independent_audit_corrected.log')
 assert read(root/'independent_audit.json')['complete']
 save(root/'audit_status.json',{'state':'complete','independent_audit_passed':True,'checker_correction':'AUDIT_CORRECTION.json','initial_failure_preserved':True})
 child=root/'refinement_interpolation'
 wait_for(child/'status.json',{'complete','skipped'},local/'refinement.log')
 phases=[root]+([child] if read(child/'status.json')['stage']=='complete' else [])
 save(root/'delivery_status.json',{'stage':'portable_inference_verification'})
 for phase in phases:
  assert read(phase/'independent_audit.json')['complete']
  if not (phase/'portable_inference_verification.json').exists():
   run(root/'verify_v3_delivered_portable.py',phase,local/('portable_refinement.log' if phase==child else 'portable_primary.log'))
  assert read(phase/'portable_inference_verification.json')['complete']
 selected=child if child in phases else root
 decision=read(selected/'decision.json');summary=read(selected/'regression_summary.json');cal=read(selected/'calibration_check.json')
 main_decision=read(root/'decision.json');config=read(selected/'inference_config.json')
 good=decision['regression_promising'];assert good==bool(config['regression_promising'])
 conditions=['png8_control','jpeg_q90_444','jpeg_q90_420'];names=['PNG','JPEG444 Q90','JPEG420 Q90']
 failures=[]
 for key,item in cal.items():
  for c in conditions:
   if item['metrics']['correct_by_condition'][c]<item['parent']['correct_by_condition'][c]:
    failures.append(f"校验集 k{key} {c}: {item['parent']['correct_by_condition'][c]}→{item['metrics']['correct_by_condition'][c]}")
  if item['metrics']['bad']>0:failures.append(f"校验集 k{key} 错误归属 {item['metrics']['bad']}")
 for row in summary:
  for c in conditions:
   if row['extension_v3']['correct_by_condition'][c]<row['parent_v1']['correct_by_condition'][c]:failures.append(f"回归 k{row['k']} {c} 正确恢复下降")
  if row['extension_v3']['bad']>row['parent_v1']['bad']:failures.append(f"回归 k{row['k']} 错误归属增加")
 if summary[0]['extension_v3']['correct_by_condition'][conditions[2]]<=summary[0]['parent_v1']['correct_by_condition'][conditions[2]]:failures.append('回归主码JPEG420没有严格提升')
 assert (not failures)==good or (selected==child and decision['blend']==[0.,0.] and not good)
 v2summary={r['k']:r for r in read(v2/'regression_summary.json')}
 result='达到本轮固定条件上的联合目标，仍为待独立验证的实验候选。' if good else '尚未达到全部联合目标；保留为实验候选，原V1发布版继续保留。'
 blend=decision.get('blend');source_step=decision.get('source_extra_step',decision.get('selected_additional_step'))
 selection=read(child/'selection_locked.json') if selected==child else {'selected_additional_step':source_step}
 lines=['# V3 最终复核','',result,'',
  '本轮从V1追加1600个训练更新，只训练layer4与bit head。每个批次同时看PNG、JPEG444和JPEG420；冻结前层、presence head与BN状态，并保留V1的PNG/444输出。生成器、软件环境、BCH和拒绝门槛均固定。','',
  f"原始训练阶段选中额外步数：{main_decision['selected_additional_step']}。0表示回退到原V1，不能称为新模型改善。",
  f"最终候选来源：{selected.name}；源训练检查点额外步数：{source_step}；layer4/bit-head融合比例：{blend}。融合阶段不增加优化器更新。",'',
  '融合是看到原始训练选择集PNG少1张后新增的补充开发阶段，单独锁定6组比例，只用selection选幅度，再对calibration和原回归队列各核验一次；它不是原始预注册训练协议的一部分。','',
  '下表为每条件1024张正样本的精确用户恢复数（用户ID完全正确且通过拒绝规则）。','',
  '| BCH+P | 图像条件 | V1 | V2 | V3候选 | V3−V1 |','|---|---|---:|---:|---:|---:|']
 table=[]
 for row in summary:
  for c,label in zip(conditions,names):
   old=row['parent_v1']['correct_by_condition'][c];new=row['extension_v3']['correct_by_condition'][c];middle=v2summary[row['k']]['extension_v2']['correct_by_condition'][c]
   lines.append(f"| (31,{row['k']}) | {label} | {old}/1024 | {middle}/1024 | {new}/1024 | {new-old:+d} |")
   table.append({'k':row['k'],'condition':c,'v1':old,'v2':middle,'v3':new,'n':1024,'delta':new-old})
 lines+=['','错误归属（含正样本认错用户及无水印图误接受，合计三条件；两种码共享负样本，不能当作独立样本相加）：']
 for row in summary:lines.append(f"- k{row['k']}: V1 {row['parent_v1']['bad']} → V3 {row['extension_v3']['bad']}。")
 lines+=['','固定校验与回归：'+('全部通过。' if good else '；'.join(failures)),'',
  '这些4096张源图及开发集已经在历史实验中观察过，属于回归复核，不能证明新独立数据上的泛化。零观察错误也不代表总体零风险。','',
  'V1、V2权重、模型包及全部冻结输入哈希均保持不变。严格环境未安装或升级任何包。1600个训练批次、冻结状态和36864个双版本决策已独立核验；补充候选另重算18432个决策。两阶段配置均已在新解压的V1便携包中显式加载，logits/bit/决策与保存结果一致。','',
  '最初的审计脚本把PyTorch 1.12.1无动量SGD状态误认为空字典，已按实际格式严格修正；原失败输出和修正依据保留于AUDIT_CORRECTION.json及audit_status_initial_failure.json。训练与结果没有改动。','',
  '交付：V3_DELIVERY.zip为最终已训练候选与文档/预测/训练记录。原始及融合阶段CANDIDATE_BUNDLE.zip分别保留，互不覆盖。所有检查点与optimizer/RNG仍在本Drive目录，未把冗余权重重复放进交付ZIP。','',
  '恢复：使用RESTORE_V3_INFERENCE.ipynb；加载到V1副本的experiments/v3并显式传--config。默认V1推理入口保持原样。环境资产从原V1私有release/offline_assets恢复，ZIP不重复包含15GB资产。','',a.folder_url]
 review='\n'.join(lines)+'\n';(root/'V3_FINAL_REVIEW_zh.md').write_text(review)
 tablehtml=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['k'],names[conditions.index(r['condition'])],f"{r['v1']}/1024",f"{r['v2']}/1024",f"{r['v3']}/1024",f"{r['delta']:+d}"])+'</tr>' for r in table)
 page='<html><meta charset="utf-8"><style>body{font:16px Arial;max-width:1040px;margin:30px auto;color:#18283a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dd;padding:9px;text-align:left}h1{color:#174775}p{line-height:1.6}.note{background:#eef3f8;padding:14px}</style><h1>WOUAF V3 · 联合鲁棒性复核</h1><p><b>'+html.escape(result)+'</b></p><p>独立版本保存 · 1600步训练完成 · V1/V2保持 · 固定环境/门槛</p><table><tr><th>k</th><th>图像条件</th><th>V1</th><th>V2</th><th>V3候选</th><th>变化</th></tr>'+tablehtml+'</table><p>'+html.escape('；'.join(failures) if failures else '固定校验与回归要求全部通过。')+'</p><p>最终候选：额外'+str(source_step)+'步；layer4/bit-head融合比例 '+html.escape(str(blend))+'。</p><p class="note">旧4096张源图是回归队列，尚非新独立泛化测试。V3_DELIVERY.zip保存已训练候选、预测和文档；完整检查点单独保留。</p><p><a href="'+html.escape(a.folder_url)+'">打开V3 Drive目录</a></p></html>'
 (root/'V3_FINAL_REVIEW_zh.html').write_text(page)
 instructions='# 在V1便携仓库加载V3候选\n\n'+result+'\n\n将V3_DELIVERY.zip解压到新V1副本的experiments/v3。保留所有原V1文件，不修改默认configs/pretrained.json。先按V1 README恢复严格环境，再运行：\n\n```bash\npython scripts/run_locked.py infer.py --config experiments/v3/inference_config.json --image YOUR_IMAGE.png --condition jpeg_q90_420 --code 11 --output outputs/v3.json\n```\n\n`--condition`描述实际图像处理条件，不会自动压缩输入；相机/JPEG文件按其实际内容输入。JPEG验证使用Pillow质量90、4:2:0或4:4:4缓存；BCH、presence和bit门槛均固定。代码16可作65,536用户对照。\n\n完整复核见V3_FINAL_REVIEW_zh.md。此ZIP是已训练模型；训练源码、原始plan和角色说明在experiment/，完整中间检查点在Drive原目录，手动选择运行训练脚本才会追加训练。\n'
 (root/'LOAD_V3_DELIVERY.md').write_text(instructions)
 release=Path(plan['parent_release_root']);bundle=selected/'CANDIDATE_BUNDLE.zip';bundle_sha=sha(bundle)
 def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
 def code(s):return {'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]}
 restore="from pathlib import Path\nimport hashlib,json,zipfile,subprocess,sys\nRELEASE=Path("+repr(str(release))+")\nV3=Path("+repr(str(selected))+")\nWORK=Path('/content/wouaf_v3_inference')\nWORK.mkdir(exist_ok=True)\nBASE=WORK/'ecc-protected-wouaf'\ndef extract_checked(path,dest,expected):\n    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected\n    with zipfile.ZipFile(path) as z:\n        assert z.testzip() is None\n        for name in z.namelist():\n            p=Path(name);assert not p.is_absolute() and '..' not in p.parts\n        z.extractall(dest)\nassert not BASE.exists(),'Use a fresh output folder; do not overwrite an existing extraction.'\nextract_checked(RELEASE/'ecc-protected-wouaf-pretrained.zip',WORK,'f0bb7cbbd7b2da23cbddf9947548c2989594d93536f7fdb7f506ebd9d1eb951b')\nVERSION=BASE/'experiments/v3'\nVERSION.mkdir(parents=True,exist_ok=False)\nextract_checked(V3/'CANDIDATE_BUNDLE.zip',VERSION,"+repr(bundle_sha)+")\nfor line in (VERSION/'BUNDLE_SHA256SUMS.txt').read_text().splitlines():\n    expected,name=line.split('  ',1);assert hashlib.sha256((VERSION/name).read_bytes()).hexdigest()==expected\n"
 nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
  md('# 恢复V3已训练候选推理\n'+result+'\n只恢复与推理，不训练。参考A100-SXM4-80GB / driver580.82.07，严格硬件/软件检查保持；环境资产需同一Drive权限。'),
  code("from google.colab import drive\ndrive.mount('/content/drive')\n"),code(restore),
  code("subprocess.run([sys.executable,str(BASE/'scripts/materialize_assets.py'),'--asset-root',str(RELEASE/'offline_assets'),'--id','frozen_environment'],check=True)\nsubprocess.run([sys.executable,str(BASE/'scripts/restore_environment.py')],check=True)\nsubprocess.run([sys.executable,str(BASE/'scripts/run_locked.py'),'--check'],check=True)\n"),
  code("example=next(r for r in json.loads((BASE/'demo/manifest.json').read_text()) if r['group']=='positive_k11')\noutput=BASE/'outputs/v3_prediction.json';output.parent.mkdir(exist_ok=True)\nsubprocess.run([sys.executable,str(BASE/'scripts/run_locked.py'),str(BASE/'infer.py'),'--config',str(VERSION/'inference_config.json'),'--image',str(BASE/example['image']),'--condition','jpeg_q90_420','--code','11','--output',str(output)],check=True)\nprint(output.read_text())\n"),
  md('恢复会校验V1和V3包SHA，原V1文件保持不变。详见V3_FINAL_REVIEW_zh.md；本候选未自动替换旧发布版。')]}
 save(root/'RESTORE_V3_INFERENCE.ipynb',nb)
 # Bundle the trained candidate plus exact protocols, predictions and a human entry point.
 archive=root/'V3_DELIVERY.zip';entries={}
 with zipfile.ZipFile(bundle) as z:
  assert z.testzip() is None
  for name in z.namelist():entries[name]=z.read(name)
 entries.pop('BUNDLE_SHA256SUMS.txt',None)
 entries['README_zh.md']=(review+'\n'+instructions).encode()
 for name in ['V3_FINAL_REVIEW_zh.md','V3_FINAL_REVIEW_zh.html','LOAD_V3_DELIVERY.md','RESTORE_V3_INFERENCE.ipynb']:
  entries[name]=(root/name).read_bytes()
 for phase in phases:
  label='refinement' if phase==child else 'primary'
  for path in phase.iterdir():
   if path.is_file() and path.suffix in ['.json','.csv','.py','.md'] and path.name not in ['delivery_status.json','V3_FINAL_REVIEW_zh.md','LOAD_V3_DELIVERY.md']:
    entries[f'experiment/{label}/{path.name}']=path.read_bytes()
 for k in [11,16]:
  for stem in ['regression_v3','calibration_fixed','selection_selected']:
   file=selected/f'{stem}_k{k}.npz'
   if file.exists():entries['predictions/'+file.name]=file.read_bytes()
 entries['reference/V1_RELEASE.json']=json.dumps({'root':str(release),'zip_sha256':'f0bb7cbbd7b2da23cbddf9947548c2989594d93536f7fdb7f506ebd9d1eb951b','offline_assets':str(release/'offline_assets'),'full_checkpoints_at':str(root)},indent=2).encode()
 manifest=''.join(f'{hashlib.sha256(value).hexdigest()}  {name}\n' for name,value in sorted(entries.items()))
 entries['V3_DELIVERY_SHA256SUMS.txt']=manifest.encode()
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for name,value in entries.items():z.writestr(name,value)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for line in z.read('V3_DELIVERY_SHA256SUMS.txt').decode().splitlines():
   expected,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
 for path,expected in plan['frozen_inputs'].items():assert sha(path)==expected,path
 delivery={'complete':True,'version':'V3','root':str(root),'selected_phase_root':str(selected),'model_version':config['version'],
  'trained_extra_steps':1600,'selected_source_extra_step':source_step,'blend':blend,'selected_model_sha256':config['weights_sha256'],
  'joint_target_met_on_observed_data':good,'independent_test':False,'release_status':'experimental_regression_candidate' if good else 'experimental_joint_target_not_met',
  'parent_versions_unchanged':True,'software_unchanged':True,'independent_audit_passed':True,'portable_inference_verified':True,
  'v1_remains_default':True,'promoted_to_main_release':False,'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
  'drive_folder_url':a.folder_url,'review':'V3_FINAL_REVIEW_zh.md','restore_entry':'RESTORE_V3_INFERENCE.ipynb','failed_criteria':failures,'table':table}
 save(root/'V3_DELIVERY.json',delivery)
 version_index=root.parent/'VERSION_INDEX.json';versions=read(version_index) if version_index.exists() else []
 old=next((v for v in versions if v.get('root')==str(root)),None)
 if old is None:old={'root':str(root),'version':root.name};versions.append(old)
 old.update({key:delivery[key] for key in ['selected_phase_root','selected_model_sha256','trained_extra_steps','release_status','independent_audit_passed','portable_inference_verified','drive_folder_url','review','restore_entry','archive_sha256']})
 old.update({'parent_weight_sha256':plan['parent_weight_sha256'],'promoted_to_main_release':False})
 save(version_index,versions)
 save(root.parent.parent/'LATEST_V3_TRAINING.json',{'root':str(root),'version':'V3','state':'complete','delivery':str(root/'V3_DELIVERY.json'),'drive_folder_url':a.folder_url,'release_status':delivery['release_status']})
 proof=['V3_DELIVERY.zip','V3_DELIVERY.json','V3_FINAL_REVIEW_zh.md','V3_FINAL_REVIEW_zh.html','RESTORE_V3_INFERENCE.ipynb','AUDIT_CORRECTION.json','independent_audit.json','audit_status_initial_failure.json']
 (root/'DELIVERY_SHA256SUMS.txt').write_text(''.join(f'{sha(root/name)}  {name}\n' for name in proof))
 save(root/'delivery_status.json',{'stage':'complete','joint_target_met':good,'archive_sha256':delivery['archive_sha256']})
 print('V3_DELIVERY_COMPLETE',json.dumps(delivery,ensure_ascii=False),flush=True)
except Exception:
 save(root/'delivery_status.json',{'stage':'failed','error':traceback.format_exc()});raise

"""Supplemental, explicitly versioned development-only weight interpolation search."""
import argparse,hashlib,importlib.util,json,shutil,sys,time,zipfile,html
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root;training=root.parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
plan=json.loads((root/'plan.json').read_text());assert sha(root/'plan.json')==json.loads((root/'protocol_lock.json').read_text())['plan_sha256']
assert sha(Path(__file__))==plan['script_sha256']
start=time.monotonic()
while True:
 state=json.loads((training/'status.json').read_text())
 if state['stage']=='failed':raise RuntimeError(state.get('error'))
 if state['stage']=='complete':break
 assert time.monotonic()-start<7200;time.sleep(10)
if json.loads((training/'decision.json').read_text())['regression_promising']:
 save(root/'status.json',{'stage':'skipped','reason':'Primary V3 already meets all frozen criteria.'});print('V3_REFINEMENT_SKIPPED');raise SystemExit(0)
original=json.loads((training/'plan.json').read_text());parent=Path(original['parent_root']);local=Path(original['parent_local'])
s=importlib.util.spec_from_file_location('refine_parent_recipe',training/'frozen_parent_recipe.py');source=importlib.util.module_from_spec(s);s.loader.exec_module(source)
source.context(parent);torch=source.configure();torch.set_grad_enabled(False)
parentplan=json.loads((parent/'plan.json').read_text());bch=source.BCH(parentplan);thresholds={11:0.,16:0.}
records,cache,targets=source.cache(parent,local);base_state=torch.load(parent/'selected_multitask.pth',map_location='cpu')
reference={};baseline={}
for k in [11,16]:
 with np.load(parent/f'selection_selected_k{k}.npz') as z:reference[k]={n:z[n].copy() for n in z.files}
 baseline[k]=source.metrics(reference[k],reference[k]['bit_gate']&(reference[k]['presence']>=0))
rows=json.loads((training/'selection_history.json').read_text())
eligible=[r for r in rows if all(r['metrics'][str(k)]['bad']==0 for k in [11,16])]
assert eligible
chosen=max(eligible,key=lambda r:(r['metrics']['11']['correct_by_condition']['jpeg_q90_420'],r['metrics']['11']['correct'],r['metrics']['16']['correct']))
step=chosen['additional_step'];checkpoint=training/f'checkpoint_extra{step:04d}.pth';trained=torch.load(checkpoint,map_location='cpu')
save(root/'source_checkpoint_lock.json',{'extra_step':step,'sha256':sha(checkpoint),'chosen_using_selection_only':True})
model=source.make_model(parentplan);model.requires_grad_(False).eval();history=[]
def rank(metrics):
 good=all(metrics[k]['bad']==0 and all(metrics[k]['correct_by_condition'][c]>=baseline[k]['correct_by_condition'][c] for c in source.CONDITIONS) for k in [11,16])
 return (good,metrics[11]['correct_by_condition']['jpeg_q90_420'],metrics[11]['correct'],metrics[16]['correct_by_condition']['jpeg_q90_420'],metrics[16]['correct'])
best_rank=rank(baseline);best=(0.,0.);shutil.copy2(parent/'selected_multitask.pth',root/'selected_multitask.pth')
save(root/'status.json',{'stage':'selection_interpolation'})
for number,(body,head) in enumerate(plan['grid'],1):
 mixed={}
 for name,value in base_state.items():
  if torch.equal(value,trained[name]):mixed[name]=value.clone();continue
  assert value.is_floating_point() and name.startswith(('backbone.layer4.','bit_head.')),name
  fraction=body if name.startswith('backbone.layer4.') else head
  mixed[name]=value+fraction*(trained[name]-value)
 model.load_state_dict(mixed,strict=True)
 raw=source.views_infer(model,records,cache,None,'selection');ds=source.decisions(records,raw,bch,True)
 metrics={k:source.metrics(d,d['bit_gate']&(d['presence']>=0)) for k,d in ds.items()};value=rank(metrics)
 if value>best_rank:
  best_rank=value;best=(body,head);torch.save(mixed,root/'selected_multitask.pth');source.save_decisions(root,'selection_selected',ds,thresholds)
 history.append({'body_fraction':body,'bit_head_fraction':head,'rank':list(value),'metrics':metrics})
 save(root/'selection_history.json',history)
 print('V3_INTERPOLATION',number,body,head,'counts',metrics[11]['correct_by_condition'],'eligible',bool(value[0]),'best',best,flush=True)
save(root/'selection_locked.json',{'complete':True,'source_extra_step':step,'body_fraction':best[0],'head_fraction':best[1],
 'selected_weight_sha256':sha(root/'selected_multitask.pth'),'no_calibration_or_regression_used_for_selection':True})
model.load_state_dict(torch.load(root/'selected_multitask.pth',map_location='cpu'),strict=True)
raw=source.views_infer(model,records,cache,None,'calibration');ds=source.decisions(records,raw,bch,True)
parentcal=json.loads((parent/'locked_presence_policy.json').read_text())['per_code'];calibration={}
for k,d in ds.items():
 m=source.metrics(d,d['bit_gate']&(d['presence']>=0));ref=parentcal[str(k)]['new_joint']
 calibration[k]={'metrics':m,'parent':ref,'passed':m['bad']==0 and all(m['correct_by_condition'][c]>=ref['correct_by_condition'][c] for c in source.CONDITIONS)}
save(root/'calibration_check.json',calibration);source.save_decisions(root,'calibration_fixed',ds,thresholds)
save(root/'status.json',{'stage':'regression'})
testrows=json.loads((parent/'future_test_image_manifest.json').read_text())
testrecords=[dict(r,item_id=f"{r['group']}:{r['sample_id']}",role='test') for r in testrows]
testcache=np.memmap(local/'future_pixels.uint8',mode='r',dtype=np.uint8,shape=(4096,3,512,512,3))
raw=source.views_infer(model,testrecords,testcache,None,'test');ds=source.decisions(testrecords,raw,bch,True);source.save_decisions(root,'regression_v3',ds,thresholds)
summary=[]
for k,d in ds.items():
 with np.load(parent/f'future_test_negative_aware_k{k}.npz') as z:old={n:z[n].copy() for n in z.files}
 assert np.array_equal(old['item_id'],d['item_id']) and np.array_equal(old['condition'],d['condition'])
 m=source.metrics(d,d['bit_gate']&(d['presence']>=0));ref=source.metrics(old,old['final_accept'])
 summary.append({'k':k,'parent_v1':ref,'extension_v3':m,'retention':{c:m['correct_by_condition'][c]/ref['correct_by_condition'][c] for c in source.CONDITIONS}})
save(root/'regression_summary.json',summary)
good=all(v['passed'] for v in calibration.values()) and all(all(r['retention'][c]>=1 for c in source.CONDITIONS) and r['extension_v3']['bad']<=r['parent_v1']['bad'] for r in summary)
good=bool(good and best!=(0.,0.) and summary[0]['extension_v3']['bad']==0 and summary[0]['extension_v3']['correct_by_condition'][source.CONDITIONS[2]]>summary[0]['parent_v1']['correct_by_condition'][source.CONDITIONS[2]])
config=json.loads((Path(original['portable_base_repo'])/'configs/pretrained.json').read_text())
config.update({'version':'v3_multiview_guard_refined','weights':'selected_multitask.pth','weights_relative_to':'configuration_directory',
 'weights_sha256':sha(root/'selected_multitask.pth'),'regression_promising':good})
save(root/'inference_config.json',config)
save(root/'decision.json',{'complete':True,'regression_promising':good,'source_extra_step':step,'blend':best,'new_optimizer_steps':0,'not_automatically_promoted':True,'independent_test':False})
# Independent portable decoder and frozen-state audit, without modifying selection or gates.
sys.path.insert(0,original['portable_base_repo']);from ecc_wouaf.core import Decoder,load_config
decoder=Decoder(load_config());count=0
selected=torch.load(root/'selected_multitask.pth',map_location='cpu');frozen=0
for name,value in base_state.items():
 if not name.startswith(('backbone.layer4.','bit_head.')) or any(x in name for x in ['running_mean','running_var','num_batches_tracked']):
  assert torch.equal(value,selected[name]);frozen+=1
for k in [11,16]:
 with np.load(root/f'regression_v3_k{k}.npz') as z:d={n:z[n].copy() for n in z.files}
 decoded=decoder.decode(d['logits'],d['presence'],k)
 for new,old in [('accepted','final_accept'),('candidate_user_id','candidate_user'),('pred_bits','pred_bits')]:assert np.array_equal(decoded[new],d[old])
 count+=len(d['logits'])
for path,value in original['frozen_inputs'].items():assert sha(path)==value,path
save(root/'independent_audit.json',{'complete':True,'saved_decisions_recomputed':count,'frozen_entries_unchanged':frozen,'parent_inputs_unchanged':True})
text='# V3补充阶段：小幅权重融合\n\n'+('达到本轮严格目标，仍需新独立队列确认。' if good else '未达到全部严格目标，仅作实验记录；V1发布版保留。')
text+='\n\n本补充阶段是在看到训练选择集的PNG一张损失后提出；它不是原始训练协议的一部分。融合网格在任何融合候选推理前锁定，只用selection角色选幅度。calibration只核验一次，旧4096源图只作回归，不宣称新独立测试。零观察错不代表总体零风险。\n\n源额外步数：'+str(step)+'；layer4/bit-head融合比例：'+str(best)+'。无新增优化器更新。\n'
for row in summary:text+=f"\nk{row['k']} V1: {row['parent_v1']['correct_by_condition']}，V3: {row['extension_v3']['correct_by_condition']}；错误归属{row['parent_v1']['bad']}→{row['extension_v3']['bad']}。\n"
(root/'README_zh.md').write_text(text)
with zipfile.ZipFile(root/'CANDIDATE_BUNDLE.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 names=['selected_multitask.pth','inference_config.json','decision.json','README_zh.md','plan.json','protocol_lock.json','source_checkpoint_lock.json','selection_locked.json','regression_summary.json','independent_audit.json']
 for name in names:z.write(root/name,arcname=name)
 z.writestr('BUNDLE_SHA256SUMS.txt','\n'.join(f'{sha(root/n)}  {n}' for n in names)+'\n')
with zipfile.ZipFile(root/'CANDIDATE_BUNDLE.zip') as z:
 assert z.testzip() is None
 for line in z.read('BUNDLE_SHA256SUMS.txt').decode().splitlines():
  expected,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==expected
save(root/'status.json',{'stage':'complete','regression_promising':good,'blend':best,'source_extra_step':step})
print('V3_REFINEMENT_COMPLETE',json.dumps({'good':good,'blend':best,'source_extra_step':step,'summary':summary}),flush=True)

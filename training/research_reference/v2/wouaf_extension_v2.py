"""Versioned V2 continuation: full-codeword objective, frozen gates, no generator updates."""
import argparse,copy,csv,hashlib,html,importlib.util,json,math,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CONDITIONS=['png8_control','jpeg_q90_444','jpeg_q90_420']
GROUPS=['positive_k11','positive_k16','generated_negative','natural_negative']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(root,resume=False):
 p=json.loads((root/'plan.json').read_text());lock=json.loads((root/'protocol_lock.json').read_text())
 assert sha(root/'plan.json')==lock['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
 parent=Path(p['parent_root']);local=Path(p['local']);local.mkdir(exist_ok=resume)
 assert p['extra_steps']==2400
 if not resume:assert not (root/'training_history.json').exists(),'This experiment has already started; use an explicit resume.'
 for path,value in p['frozen_inputs'].items():assert sha(path)==value,path
 original=json.loads((parent/'plan.json').read_text());source=module('v2_frozen_parent_recipe',root/'frozen_parent_recipe.py')
 source.context(parent);torch=source.configure(True);torch.manual_seed(p['seed'])
 records,mmap,targets=source.cache(parent,Path(p['parent_local']));source.assert_data(parent)
 assert sha(Path(p['parent_local'])/'development_pixels.uint8')==json.loads((parent/'preparation.json').read_text())['cache_sha256']
 roles={r['item_id']:r['role'] for r in records}
 assert all(roles[key]==value for key,value in p['role_lock'].items()) and len(roles)==4456
 content_roles={};user_roles={}
 for r in records:
  content_roles.setdefault(r['content_key'],set()).add(r['role'])
  if r['user_id']>=0:user_roles.setdefault((r['k'],r['user_id']),set()).add(r['role'])
 assert all(len(s)==1 for s in list(content_roles.values())+list(user_roles.values()))
 bch=source.BCH(original);policy=json.loads((parent/'locked_presence_policy.json').read_text())
 thresholds={k:policy['per_code'][str(k)]['threshold'] for k in [11,16]}
 assert thresholds=={11:0.,16:0.}
 save(root/'fixed_policy.json',{'bit_thresholds':bch.tau,'presence_thresholds':thresholds,'source_policy_sha256':sha(parent/'locked_presence_policy.json'),
  'no_gate_fitting_or_retuning_in_v2':True})
 def state(stage,**extra):save(root/'status.json',{'stage':stage,'updated_unix':time.time(),**extra})
 state('prepare_teacher_and_codebooks')
 model=source.make_model(original);model.load_state_dict(torch.load(parent/'selected_multitask.pth',map_location='cpu'),strict=True)
 initial={name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
 frozen=[name for name,v in model.named_parameters() if not v.requires_grad]
 buffers=[n for n in initial if any(v in n for v in ['running_mean','running_var','num_batches_tracked'])]
 parent_raw={};parent_metrics={}
 for k in [11,16]:
  with np.load(parent/f'selection_selected_k{k}.npz') as z:parent_raw[k]={n:z[n].copy() for n in z.files}
  parent_metrics[k]=source.metrics(parent_raw[k],parent_raw[k]['bit_gate']&(parent_raw[k]['presence']>=thresholds[k]))
  assert parent_metrics[k]['bad']==0
 save(root/'parent_selection_metrics.json',parent_metrics)
 # Fit-only frozen teacher outputs from the actual V1 selected model, not the older +300 donor.
 with torch.no_grad():teacher_raw=source.views_infer(model,records,mmap,None,'fit')
 teacher_bits=np.zeros((len(records),3,32),np.float32);teacher_presence=np.zeros((len(records),3),np.float32);teacher_valid=np.zeros((len(records),3),bool)
 for c,condition in enumerate(CONDITIONS):
  m=teacher_raw['condition']==condition;ix=teacher_raw['index'][m]
  teacher_bits[ix,c]=teacher_raw['logits'][m];teacher_presence[ix,c]=teacher_raw['presence'][m];teacher_valid[ix,c]=True
 np.savez_compressed(root/'fit_parent_teacher.npz',item_id=np.array([r['item_id'] for r in records]),logits=teacher_bits,presence=teacher_presence,valid=teacher_valid)
 assert teacher_valid.sum()==6684
 books={}
 for k in [11,16]:
  users=np.arange(2**k,dtype=np.uint32);messages=((users[:,None]>>np.arange(k-1,-1,-1))&1).astype(np.uint8)
  generator=bch.books[k][3];word=messages@generator%2;full=np.column_stack([word,word.sum(axis=1)%2]).astype(np.uint8)
  assert np.array_equal(messages,full[:,:k]) and np.all(full.sum(axis=1)%2==0)
  for i,r in enumerate(records):
   if r['group']==f'positive_k{k}':assert np.array_equal(full[r['user_id']],targets['gt_bits'][i])
  books[k]=torch.from_numpy(full.astype(np.float32)).cuda()
 # Verify the new CE objective against its direct full-codebook expression before any optimizer update.
 ix=np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='positive_k11'][:4])
 logits=torch.from_numpy(teacher_bits[ix,2].copy()).cuda();user=torch.tensor([records[i]['user_id'] for i in ix],device='cuda')
 scores=logits@books[11].T;loss=F.cross_entropy(scores,user)
 direct=(torch.logsumexp(scores,1)-scores[torch.arange(len(ix),device='cuda'),user]).mean()
 assert torch.allclose(loss,direct,atol=1e-5,rtol=1e-5)
 # Reproduce twelve saved parent views through the exact uint8 cache used for this continuation.
 controls=[]
 model.eval()
 with torch.no_grad():
  for condition in CONDITIONS:
   d=parent_raw[11];picked=np.flatnonzero((d['condition']==condition)&(d['group']=='positive_k11'))[:4]
   rows=d['index'][picked];c=CONDITIONS.index(condition)
   values,pres=model(source.normalized(torch,mmap[rows,c]));values=values.cpu().numpy();pres=pres.cpu().numpy()
   assert np.allclose(values,d['logits'][picked],atol=1e-4,rtol=1e-4) and np.allclose(pres,d['presence'][picked],atol=1e-4,rtol=1e-4)
   assert np.array_equal(torch.from_numpy(values).sigmoid().numpy()>.5,d['pred_bits'][picked])
   controls.append({'condition':condition,'views':4,'max_logit_difference':float(np.abs(values-d['logits'][picked]).max())})
 save(root/'preflight.json',{'complete':True,'teacher_views':6684,'teacher_roles':['fit'],'parent_reference_views':controls,
  'full_codebook_sizes':{'11':2048,'16':65536},'codeword_ce_matches_direct_expression':True,'no_optimizer_updates_before_this_check':True})
 print('V2_PREFLIGHT_PASSED',flush=True)
 pools={g:np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']==g]) for g in GROUPS}
 paired={r['sample_id']:i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='generated_negative'}
 groups=[{'params':[v for n,v in model.named_parameters() if n.startswith(prefix) and v.requires_grad],'lr':rate,'initial_rate':rate}
  for prefix,rate in [('backbone.layer3.',1e-6),('backbone.layer4.',2.5e-6),('bit_head.',5e-6),('presence_head.',2e-5)]]
 optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);random=np.random.default_rng(p['seed']);history=[];assessments=[]
 trainable=[v for v in model.parameters() if v.requires_grad];bce=nn.BCEWithLogitsLoss()
 def ranking(measured):
  retain={k:{c:measured[k]['correct_by_condition'][c]/max(1,parent_metrics[k]['correct_by_condition'][c]) for c in CONDITIONS} for k in [11,16]}
  eligible=all(measured[k]['bad']==0 for k in [11,16]) and min(retain[11].values())>=.98 and min(retain[16].values())>=.95
  return (eligible,measured[11]['correct_by_condition']['jpeg_q90_420'],measured[11]['correct'],
   measured[16]['correct_by_condition']['jpeg_q90_420'],measured[16]['correct']),retain
 best_rank,_=ranking(parent_metrics);best_step=0;assert best_rank[0]
 start_step=1;prior_elapsed=0.
 if resume:
  previous=torch.load(root/'last_resumable_checkpoint.pth',map_location='cpu')
  assert sha(root/'selected_multitask.pth')==previous['best_weight_sha256']
  model.load_state_dict(previous['state_dict'],strict=True);optimizer.load_state_dict(previous['optimizer'])
  random.bit_generator.state=previous['numpy_rng'];torch.set_rng_state(previous['torch_rng']);torch.cuda.set_rng_state_all(previous['cuda_rng'])
  best_step=previous['best_additional_step'];best_rank=tuple(previous['best_rank']);start_step=previous['step']+1;prior_elapsed=previous['elapsed_seconds']
  history=json.loads((root/'training_history.json').read_text());assessments=json.loads((root/'selection_history.json').read_text())
 else:shutil.copy2(parent/'selected_multitask.pth',root/'selected_multitask.pth')
 started=time.perf_counter()-prior_elapsed;state('training',step=start_step-1,total=p['extra_steps'],best_additional_step=best_step)
 for step in range(start_step,p['extra_steps']+1):
  assert time.perf_counter()-started<p['training_timeout_seconds'],'Bounded training timeout; retain saved checkpoints.'
  positive=np.concatenate([random.choice(pools[g],4,replace=False) for g in GROUPS[:2]])
  matched=np.array([paired[records[i]['sample_id']] for i in positive]);ix=np.concatenate([positive,matched,random.choice(pools['generated_negative'],4,replace=False),random.choice(pools['natural_negative'],4,replace=False)])
  c=[0,2,1,2,2][(step-1)%5]
  assert len(ix)==24 and all(records[i]['role']=='fit' for i in ix) and teacher_valid[ix,c].all()
  model.eval();model.presence_head.train();optimizer.zero_grad(set_to_none=True)
  bits,presence=model(source.normalized(torch,mmap[ix,c]));gt=torch.from_numpy(targets['gt_bits'][positive].copy()).float().cuda()
  loss_bits=bce(bits[:8],gt)
  loss_presence=.5*bce(presence[:8],torch.ones_like(presence[:8]))+.25*bce(presence[8:20],torch.zeros_like(presence[8:20]))+.25*bce(presence[20:],torch.zeros_like(presence[20:]))
  neutral=.5*bce(bits[8:20],torch.full_like(bits[8:20],.5))+.5*bce(bits[20:],torch.full_like(bits[20:],.5))
  teacher=torch.from_numpy(teacher_bits[positive,c].copy()).cuda();retain=(bits[:8]-teacher).square().mean()
  old_presence=torch.from_numpy(teacher_presence[ix,c].copy()).cuda();presence_retain=(presence.clamp(-10,10)-old_presence.clamp(-10,10)).square().mean()
  paired_loss=F.softplus(1-presence[:8]+presence[8:16]).mean();ce=[]
  for k,begin in [(11,0),(16,4)]:
   users=torch.tensor([records[i]['user_id'] for i in positive[begin:begin+4]],device='cuda')
   ce.append(F.cross_entropy(bits[begin:begin+4]@books[k].T,users)/32)
  codeword=.5*(ce[0]+ce[1]);retention_weight=.01 if c==2 else .1
  loss=2*loss_bits+loss_presence+.05*neutral+retention_weight*retain+.01*presence_retain+.2*paired_loss+codeword
  assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)
  factor=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0.,(step-100)/(p['extra_steps']-100)))))
  for g in optimizer.param_groups:g['lr']=g['initial_rate']*factor
  optimizer.step()
  history.append({'additional_step':step,'condition':CONDITIONS[c],'fit_indices':ix.tolist(),'loss':float(loss.item()),'bit_BCE':float(loss_bits.item()),
   'presence_BCE':float(loss_presence.item()),'codeword_CE_per_bit':float(codeword.item()),'parent_bit_retention':float(retain.item()),'parent_presence_retention':float(presence_retain.item())})
  if step%100==0:
   save(root/'training_history.json',history);state('training',step=step,total=p['extra_steps'],best_additional_step=best_step,elapsed_seconds=time.perf_counter()-started)
   print('V2_TRAIN',step,p['extra_steps'],'loss',round(float(loss.item()),5),'codeword',round(float(codeword.item()),5),'elapsed',round(time.perf_counter()-started,1),flush=True)
  if step%400==0:
   raw=source.views_infer(model,records,mmap,None,'selection');ds=source.decisions(records,raw,bch,True)
   measured={k:source.metrics(d,d['bit_gate']&(d['presence']>=thresholds[k])) for k,d in ds.items()}
   rank,retention=ranking(measured);snapshot={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
   torch.save(snapshot,root/f'checkpoint_extra{step:04d}.pth')
   if rank>best_rank:
    best_step=step;best_rank=rank;torch.save(snapshot,root/'selected_multitask.pth');source.save_decisions(root,'selection_selected',ds,thresholds)
   assessments.append({'additional_step':step,'rank':list(rank),'metrics':measured,'retention_vs_v1':retention,'best_additional_step':best_step})
   save(root/'selection_history.json',assessments)
   torch.save({'state_dict':snapshot,'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':random.bit_generator.state,
    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best_additional_step':best_step,'best_rank':list(best_rank),
    'best_weight_sha256':sha(root/'selected_multitask.pth'),'elapsed_seconds':time.perf_counter()-started},root/'last_resumable_checkpoint.pth')
   print('V2_SELECTION',json.dumps({'extra_step':step,'primary':measured[11]['correct_by_condition'],'bad11':measured[11]['bad'],
    'secondary':measured[16]['correct_by_condition'],'bad16':measured[16]['bad'],'eligible':bool(rank[0]),'best_extra_step':best_step}),flush=True)
 for n in frozen+buffers:assert torch.equal(model.state_dict()[n].cpu(),initial[n]),n
 save(root/'selection_locked.json',{'complete':True,'additional_steps_completed':p['extra_steps'],'selected_additional_step':best_step,
  'selected_weight_sha256':sha(root/'selected_multitask.pth'),'parent_weight_sha256':p['parent_weight_sha256'],
  'rank':list(best_rank),'test_and_calibration_not_used_for_selection':True,'frozen_parameter_count':len(frozen),'batchnorm_buffers_unchanged':True})
 print('V2_TRAINING_COMPLETE',best_step,flush=True)

 # Selection is now locked; held development calibration is a check of the SAME gates, not a new fit.
 state('fixed_policy_calibration_check',step=p['extra_steps']);model.load_state_dict(torch.load(root/'selected_multitask.pth',map_location='cpu'),strict=True)
 raw=source.views_infer(model,records,mmap,None,'calibration');ds=source.decisions(records,raw,bch,True);calibration={}
 for k,d in ds.items():
  measured=source.metrics(d,d['bit_gate']&(d['presence']>=thresholds[k]));reference=policy['per_code'][str(k)]['new_joint']
  retention={c:measured['correct_by_condition'][c]/max(1,reference['correct_by_condition'][c]) for c in CONDITIONS}
  calibration[k]={'fixed_threshold':thresholds[k],'metrics':measured,'parent':reference,'retention':retention,
   'passed':measured['bad']==0 and min(retention.values())>=(.98 if k==11 else .95)}
 source.save_decisions(root,'calibration_fixed',ds,thresholds);save(root/'calibration_check.json',calibration)

 # Previously observed 4096-source cohort: explicitly a regression benchmark, not independent evidence.
 state('regression_evaluation',step=p['extra_steps']);testrows=json.loads((parent/'future_test_image_manifest.json').read_text())
 testrecords=[dict(r,item_id=f"{r['group']}:{r['sample_id']}",role='test') for r in testrows]
 testcache=np.memmap(Path(p['parent_local'])/'future_pixels.uint8',mode='r',dtype=np.uint8,shape=(4096,3,512,512,3))
 assert len(testrows)==4096
 for r in testrows:assert sha(r['path'])==r['sha256']
 raw=source.views_infer(model,testrecords,testcache,None,'test');tests=source.decisions(testrecords,raw,bch,True)
 source.save_decisions(root,'regression_v2',tests,thresholds)
 allrows=[];summary=[];paired_results=[]
 for k,d in tests.items():
  measured=source.metrics(d,d['bit_gate']&(d['presence']>=thresholds[k]))
  with np.load(parent/f'future_test_negative_aware_k{k}.npz') as z:old={n:z[n].copy() for n in z.files}
  assert np.array_equal(old['item_id'],d['item_id']) and np.array_equal(old['condition'],d['condition'])
  baseline=source.metrics(old,old['final_accept']);newaccept=d['bit_gate']&(d['presence']>=thresholds[k])
  oldcorrect=old['final_accept']&old['correct_candidate'];newcorrect=newaccept&d['correct_candidate']
  gains=int((newcorrect&~oldcorrect).sum());losses=int((oldcorrect&~newcorrect).sum());assert gains-losses==measured['correct']-baseline['correct']
  paired_results.append({'k':k,'new_correct_only':gains,'parent_correct_only':losses,'net_correct':gains-losses})
  retention={c:measured['correct_by_condition'][c]/max(1,baseline['correct_by_condition'][c]) for c in CONDITIONS}
  summary.append({'k':k,'parent_v1':baseline,'extension_v2':measured,'retention':retention,'independent_test':False})
  for label,m in [('parent_v1',baseline),('extension_v2',measured)]:
   for row in m['cells']:allrows.append({'version':label,'k':k,**row})
 save(root/'regression_summary.json',summary);save(root/'paired_correct_changes.json',paired_results)
 with (root/'regression_cells.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)

 # Independently re-decode saved logits; replay real image views to catch export/order errors.
 checked=0
 for k in [11,16]:
  with np.load(root/f'regression_v2_k{k}.npz') as z:v={n:z[n].copy() for n in z.files}
  decoded=bch.decide(v['logits'],k,True)
  for name in ['candidate_user','ecc','bit_gate','distance32','pred_bits']:assert np.array_equal(decoded[name],v[name]),name
  assert np.array_equal(v['final_accept'],v['bit_gate']&(v['presence']>=thresholds[k]));checked+=len(v['logits'])
 replay=0;maximum=0.;model.eval()
 with torch.no_grad():
  for c,condition in enumerate(CONDITIONS):
   for group in GROUPS:
    indices=np.array([i for i,r in enumerate(testrecords) if r['group']==group][:4]);bits,pres=model(source.normalized(torch,testcache[indices,c]))
    k=16 if group=='positive_k16' else 11;d=tests[k]
    for j,i in enumerate(indices):
     key=testrecords[i]['item_id'];found=np.flatnonzero((d['item_id']==key)&(d['condition']==condition));assert len(found)==1;z=int(found[0])
     value=bits[j].cpu().numpy();assert np.allclose(value,d['logits'][z],atol=1e-4,rtol=1e-4)
     assert np.array_equal(torch.from_numpy(value).sigmoid().numpy()>.5,d['pred_bits'][z]);maximum=max(maximum,float(np.abs(value-d['logits'][z]).max()));replay+=1
 primary,secondary=summary
 promising=bool(best_step>0 and all(v['passed'] for v in calibration.values()) and primary['extension_v2']['bad']==0
  and min(primary['retention'].values())>=.98 and primary['extension_v2']['correct']>=primary['parent_v1']['correct']
  and primary['extension_v2']['correct_by_condition']['jpeg_q90_420']>primary['parent_v1']['correct_by_condition']['jpeg_q90_420']
  and secondary['extension_v2']['bad']<=secondary['parent_v1']['bad'] and min(secondary['retention'].values())>=.95)
 save(root/'validation.json',{'complete':True,'decisions_recomputed':checked,'source_images':4096,'image_views':12288,'real_views_replayed':replay,
  'maximum_replay_logit_difference':maximum,'fixed_gates_unchanged':True,'no_independent_test_claim':True,'regression_promising':promising})
 for path,value in p['frozen_inputs'].items():assert sha(path)==value,path
 assert sha(parent/'selected_multitask.pth')==p['parent_weight_sha256']
 save(root/'decision.json',{'version':'v2_codeword_jpeg420','complete':True,'selected_additional_step':best_step,
  'extra_steps_completed':p['extra_steps'],'regression_promising':promising,'not_automatically_promoted':True,
  'requires_fresh_independent_cohort_before_generalization_claim':True,'parent_release_unchanged':True})
 title='V2 扩展训练：完整码字目标＋JPEG420恢复'
 table='<table><tr><th>版本</th><th>码</th><th>PNG正确/1024</th><th>JPEG444正确/1024</th><th>JPEG420正确/1024</th><th>错误归属总数</th></tr>'
 for entry in summary:
  for label in ['parent_v1','extension_v2']:
   m=entry[label];cells=[label,entry['k']]+[m['correct_by_condition'][c] for c in CONDITIONS]+[m['bad']]
   table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>'
 table+='</table>'
 verdict='通过预先固定的回归候选要求，保存为独立V2候选；V1发布包保持原样。' if promising else '未通过全部预先固定的回归候选要求；V1仍是保留发布版本，V2结果仅作为实验记录。'
 (root/'RESULTS_zh.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px Arial;max-width:1150px;margin:28px auto;color:#18364a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #abc;padding:10px}p{line-height:1.6}</style><h1>'+title+'</h1><p>'+verdict+'</p>'+table+
  '<p>本轮追加2400步，所选额外步数 '+str(best_step)+'。固定baseline-3000生成器、软件环境、BCH和拒绝门槛；只在原fit角色更新权重，selection选点，calibration作固定规则核验。4096源图是已观察过的回归队列，不能视作新独立测试。错误归属包含正样本认错用户与无水印图被接受；同图压缩变体不是独立样本。</p></html>')
 (root/'README_zh.md').write_text('# '+title+'\n\n'+verdict+'\n\n父版本：'+str(parent)+'\n父模型SHA256：'+p['parent_weight_sha256']+'\n\n此目录是独立实验版本，不修改之前GitHub准备发布包。模型为同一双头架构，selected_multitask.pth为选中权重；plan/protocol_lock记录预先固定的配方和规则，training_history/selection_history记录过程，last_resumable_checkpoint包含optimizer/RNG。\n\n目标：在原双头loss基础上加入完整BCH码本交叉熵（除以32），并使用当前V1作为fit-only教师。真实图像条件按PNG/JPEG420/JPEG444/JPEG420/JPEG420循环；分组LR为1e-6/2.5e-6/5e-6/2e-5。2,400追加步，100 warmup + cosine；BN统计和前层冻结。\n\n只有selection同时保持k11>=98%、k16>=95%的每条件正确数且两码都无坏接受时，才按JPEG420正确数等固定顺序选择；父模型也作为step0备选。BCH/bit/presence门槛全程固定，测试不用于选点/调门槛。calibration角色已在历史研究使用，因此此次也不是全新独立数据。\n\n当前回归表见RESULTS_zh.html、regression_cells.csv；逐样本GT、logits、bits、presence和决策见regression_v2_k*.npz。观察到零错不等于真实零风险。未自动替换发布模型；需要新队列进一步验证泛化。\n')
 portable_config=json.loads((Path(p['portable_base_repo'])/'configs/pretrained.json').read_text())
 portable_config.update({'weights':'selected_multitask.pth','weights_relative_to':'configuration_directory','weights_sha256':sha(root/'selected_multitask.pth'),
  'version':'v2_codeword_jpeg420','parent_version':'v1_negative_aware_2400','trained':True,'regression_promising':promising})
 save(root/'inference_config.json',portable_config)
 (root/'LOAD_V2_IN_V1.md').write_text('# 在便携推理包中显式加载V2\n\n将本候选包解压到V1便携仓库中的 experiments/v2/，不替换 models/ 下的V1权重或 configs/pretrained.json。然后在V1仓库根目录运行：\n\n```bash\npython scripts/run_locked.py infer.py --config experiments/v2/inference_config.json --image YOUR_512_IMAGE.png --code 11 --output outputs/v2_prediction.json\n```\n\ninference_config.json引用同目录的selected_multitask.pth并校验完整SHA256；BCH及环境配置使用原V1根目录的同一套冻结文件。默认不带--config仍加载V1。归属结果需同时保留所用配置和权重hash以区分版本。此包是独立研究候选，尚未自动替换V1发布版；regression_promising不等于全新独立评估通过。\n')
 files=[x for x in sorted(root.iterdir()) if x.is_file() and x.name not in ['pipeline.log','status.json','SHA256SUMS.txt','CANDIDATE_BUNDLE.zip','BUNDLE_SHA256SUMS.txt']]
 (root/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(x)}  {x.name}' for x in files)+'\n')
 for line in (root/'SHA256SUMS.txt').read_text().splitlines():
  expected,name=line.split('  ',1);assert sha(root/name)==expected,name
 # A compact self-describing trained candidate bundle is separate from the unchanged V1 GitHub ZIP.
 bundle_names=['selected_multitask.pth','inference_config.json','LOAD_V2_IN_V1.md','fixed_policy.json','plan.json','protocol_lock.json','selection_locked.json','decision.json','regression_summary.json',
   'regression_cells.csv','RESULTS_zh.html','README_zh.md','validation.json','wouaf_extension_v2.py','frozen_parent_recipe.py','preflight.json']
 (root/'BUNDLE_SHA256SUMS.txt').write_text('\n'.join(f'{sha(root/name)}  {name}' for name in bundle_names)+'\n')
 with zipfile.ZipFile(local/'CANDIDATE_BUNDLE.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for name in bundle_names+['BUNDLE_SHA256SUMS.txt']:
   z.write(root/name,arcname=name)
 shutil.copy2(local/'CANDIDATE_BUNDLE.zip',root/'CANDIDATE_BUNDLE.zip')
 with zipfile.ZipFile(root/'CANDIDATE_BUNDLE.zip') as z:
  assert z.testzip() is None and hashlib.sha256(z.read('selected_multitask.pth')).hexdigest()==sha(root/'selected_multitask.pth')
 state('complete',step=p['extra_steps'],best_additional_step=best_step,regression_promising=promising,bundle_sha256=sha(root/'CANDIDATE_BUNDLE.zip'))
 save(root.parent.parent/'LATEST_EXTENSION_TRAINING.json',{'root':str(root),'version':'v2_codeword_jpeg420','state':'complete',
  'regression_promising':promising,'selected_additional_step':best_step,'parent_release_unchanged':True})
 print('V2_COMPLETE',json.dumps({'root':str(root),'best_extra_step':best_step,'regression_promising':promising,'summary':summary}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 try:run(args.root,args.resume)
 except Exception:
  save(args.root/'status.json',{'stage':'failed','error':traceback.format_exc(),'updated_unix':time.time()});raise

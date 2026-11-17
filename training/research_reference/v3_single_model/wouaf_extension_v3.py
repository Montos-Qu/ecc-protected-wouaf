"""Versioned V3 continuation: full-codeword objective, frozen gates, no generator updates."""
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


def balanced_presence(bce,p):
 return .5*bce(p[:4],torch.ones_like(p[:4]))+.25*bce(p[4:10],torch.zeros_like(p[4:10]))+.25*bce(p[10:],torch.zeros_like(p[10:]))
def uncertainty(bce,bits):
 return .5*bce(bits[4:10],torch.full_like(bits[4:10],.5))+.5*bce(bits[10:],torch.full_like(bits[10:],.5))
def protect(g,a,b):
 # Closest feasible vector for two first-order no-increase halfspaces.
 dot=lambda x,y:float(torch.sum(x*y,dtype=torch.float64).item())
 n0,n1,c=dot(a,a),dot(b,b),dot(a,b);d0,d1=dot(g,a),dot(g,b);tol=1e-8
 candidates=[]
 def add(x,y):
  if x>=-tol and y>=-tol and d0+x*n0+y*c>=-tol and d1+x*c+y*n1>=-tol:
   candidates.append((x*x*n0+2*x*y*c+y*y*n1,max(0.,x),max(0.,y)))
 add(0.,0.)
 if n0>1e-20:add(-d0/n0,0.)
 if n1>1e-20:add(0.,-d1/n1)
 determinant=n0*n1-c*c
 if determinant>max(1e-30,1e-12*n0*n1):add((-n1*d0+c*d1)/determinant,(c*d0-n0*d1)/determinant)
 if candidates:
  _,x,y=min(candidates);out=g+x*a+y*b;fallback=False
 else:out=torch.zeros_like(g);x=y=0.;fallback=True
 after=[dot(out,a),dot(out,b)]
 assert min(after)>=-1e-6 and torch.isfinite(out).all()
 return out,{'before':[d0,d1],'after':after,'multipliers':[x,y],'zero_fallback':fallback}

def run(root,resume=False):
 p=json.loads((root/'plan.json').read_text());lock=json.loads((root/'protocol_lock.json').read_text())
 assert sha(root/'plan.json')==lock['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
 parent=Path(p['parent_root']);local=Path(p['local']);local.mkdir(exist_ok=resume)
 assert p['extra_steps']==1600
 if not resume:assert not (root/'training_history.json').exists(),'This experiment has already started; use an explicit resume.'
 for path,value in p['frozen_inputs'].items():assert sha(path)==value,path
 original=json.loads((parent/'plan.json').read_text());source=module('v3_frozen_parent_recipe',root/'frozen_parent_recipe.py')
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
  'no_gate_fitting_or_retuning_in_v3':True})
 def state(stage,**extra):save(root/'status.json',{'stage':stage,'updated_unix':time.time(),**extra})
 state('prepare_teacher_and_codebooks')
 model=source.make_model(original);model.load_state_dict(torch.load(parent/'selected_multitask.pth',map_location='cpu'),strict=True)
 for n,v in model.named_parameters():v.requires_grad_(n.startswith(('backbone.layer4.','bit_head.')))
 initial={name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
 frozen=[name for name,v in model.named_parameters() if not v.requires_grad]
 buffers=[n for n in initial if any(v in n for v in ['running_mean','running_var','num_batches_tracked'])]
 parent_raw={};parent_metrics={}
 for k in [11,16]:
  with np.load(parent/f'selection_selected_k{k}.npz') as z:parent_raw[k]={n:z[n].copy() for n in z.files}
  parent_metrics[k]=source.metrics(parent_raw[k],parent_raw[k]['bit_gate']&(parent_raw[k]['presence']>=thresholds[k]))
  assert parent_metrics[k]['bad']==0
 save(root/'parent_selection_metrics.json',parent_metrics)
 # Reuse the immutable fit-only V1 teacher cache, and compute V2 hints on fit images only.
 with np.load(p['v1_teacher_cache']) as z:
  assert np.array_equal(z['item_id'],[r['item_id'] for r in records])
  teacher_bits=z['logits'].copy();teacher_presence=z['presence'].copy();teacher_valid=z['valid'].copy()
 assert teacher_valid.sum()==6684
 shutil.copy2(p['v1_teacher_cache'],root/'fit_parent_teacher.npz')
 teacher2=source.make_model(original);teacher2.load_state_dict(torch.load(Path(p['v2_root'])/'selected_multitask.pth',map_location='cpu'),strict=True)
 teacher2.requires_grad_(False).eval();fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
 hint=np.zeros((len(records),32),np.float32);hint_presence=np.zeros(len(records),np.float32)
 with torch.no_grad():
  for begin in range(0,len(fit),8):
   ix=fit[begin:begin+8];bits,pres=teacher2(source.normalized(torch,mmap[ix,2]));hint[ix]=bits.cpu().numpy();hint_presence[ix]=pres.cpu().numpy()
 del teacher2;torch.cuda.empty_cache()
 hints_valid=np.zeros(len(records),bool)
 for k in [11,16]:
  ix=np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']==f'positive_k{k}'])
  d=bch.decide(hint[ix],k,True);truth=targets['gt_bits'][ix].astype(float)
  oldloss=np.logaddexp(0,-(2*truth-1)*teacher_bits[ix,2]).mean(1)
  newloss=np.logaddexp(0,-(2*truth-1)*hint[ix]).mean(1)
  hints_valid[ix]=d['bit_gate']&(hint_presence[ix]>=0)&(d['candidate_user']==np.array([records[i]['user_id'] for i in ix]))&(newloss<oldloss)
 np.savez_compressed(root/'v2_fit_jpeg420_hints.npz',logits=hint,valid=hints_valid,item_id=np.array([r['item_id'] for r in records]))
 save(root/'hint_summary.json',{'fit_positive_hints':int(hints_valid.sum()),'only_correctly_decoded_V2_with_lower_GT_BCE':True,'no_selection_calibration_or_test_hints':True})
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
 for gv,av,bv,expected in [([-1.,-1.],[1.,0.],[0.,1.],[0.,0.]),([-1.,2.],[1.,0.],[0.,1.],[0.,2.]),([2.,1.],[1.,0.],[-1.,0.],[0.,1.])]:
  got,_=protect(torch.tensor(gv),torch.tensor(av),torch.tensor(bv));assert torch.allclose(got,torch.tensor(expected),atol=1e-6)
 save(root/'projection_preflight.json',{'passed':True,'orthogonal_parallel_and_opposed_controls':True,'first_order_not_heldout_guarantee':True})
 print('V3_PREFLIGHT_PASSED',flush=True)
 pools={g:np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']==g]) for g in GROUPS}
 paired={r['sample_id']:i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='generated_negative'}
 groups=[{'params':[v for v in model.parameters() if v.requires_grad],'lr':p['learning_rate'],'initial_rate':p['learning_rate']}]
 optimizer=torch.optim.SGD(groups,lr=p['learning_rate'],momentum=0.,weight_decay=0.)
 random=np.random.default_rng(p['seed']);history=[];assessments=[]
 trainable=[v for v in model.parameters() if v.requires_grad];bce=nn.BCEWithLogitsLoss()
 def ranking(measured):
  retain={k:{c:measured[k]['correct_by_condition'][c]/max(1,parent_metrics[k]['correct_by_condition'][c]) for c in CONDITIONS} for k in [11,16]}
  eligible=all(measured[k]['bad']==0 for k in [11,16]) and all(retain[k][c]>=1. for k in [11,16] for c in CONDITIONS[:2]) and all(retain[k][CONDITIONS[2]]>=1. for k in [11,16])
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
  positive=np.concatenate([random.choice(pools[g],2,replace=False) for g in GROUPS[:2]])
  matched=np.array([paired[records[i]['sample_id']] for i in positive]);ix=np.concatenate([positive,matched,random.choice(pools['generated_negative'],2,replace=False),random.choice(pools['natural_negative'],2,replace=False)])
  assert len(ix)==12 and all(records[i]['role']=='fit' for i in ix) and teacher_valid[ix].all()
  model.eval();optimizer.zero_grad(set_to_none=True)
  gt=torch.from_numpy(targets['gt_bits'][positive].copy()).float().cuda();sign=2*gt-1
  clean_grad=[];clean_values=[]
  for c in [0,1]:
   bits,presence=model(source.normalized(torch,mmap[ix,c]))
   teacher=torch.from_numpy(teacher_bits[positive,c].copy()).cuda();old_presence=torch.from_numpy(teacher_presence[ix,c].copy()).cuda()
   retain=(bits[:4]-teacher).square().mean()
   margin=F.relu((sign*teacher).clamp(0,8)-sign*bits[:4]).square().mean()
   presence_retain=(presence.clamp(-10,10)-old_presence.clamp(-10,10)).square().mean()
   clean_loss=2*bce(bits[:4],gt)+retain+margin+.2*presence_retain+.25*balanced_presence(bce,presence)+.05*uncertainty(bce,bits)
   clean_values.append(float(clean_loss.item()))
   clean_grad.append(torch.cat([v.detach().reshape(-1) for v in torch.autograd.grad(clean_loss,trainable)]))
  c=2;bits,presence=model(source.normalized(torch,mmap[ix,c]));loss_bits=bce(bits[:4],gt);loss_presence=balanced_presence(bce,presence)
  neutral=uncertainty(bce,bits);target=teacher_bits[positive,2].copy();use_hint=hints_valid[positive];target[use_hint]=hint[positive[use_hint]]
  teacher=torch.from_numpy(target).cuda();retain=(bits[:4]-teacher).square().mean()
  old_presence=torch.from_numpy(teacher_presence[ix,2].copy()).cuda();presence_retain=(presence.clamp(-10,10)-old_presence.clamp(-10,10)).square().mean()
  paired_loss=F.softplus(1-presence[:4]+presence[4:8]).mean();ce=[]
  for k,begin in [(11,0),(16,2)]:
   users=torch.tensor([records[i]['user_id'] for i in positive[begin:begin+2]],device='cuda')
   ce.append(F.cross_entropy(bits[begin:begin+2]@books[k].T,users)/32)
  codeword=.5*(ce[0]+ce[1]);loss=2*loss_bits+loss_presence+.05*neutral+.05*retain+.1*presence_retain+.2*paired_loss+codeword
  assert torch.isfinite(loss)
  jpeg_grad=torch.cat([v.detach().reshape(-1) for v in torch.autograd.grad(loss,trainable)])
  combined=jpeg_grad+.5*(clean_grad[0]+clean_grad[1]);projected,projection=protect(combined,*clean_grad)
  offset=0
  for parameter in trainable:
   size=parameter.numel();parameter.grad=projected[offset:offset+size].view_as(parameter);offset+=size
  norm=nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)
  factor=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0.,(step-100)/(p['extra_steps']-100)))))
  for g in optimizer.param_groups:g['lr']=g['initial_rate']*factor
  optimizer.step()
  history.append({'additional_step':step,'condition':'paired_three_views','fit_indices':ix.tolist(),'loss':float(loss.item()),'bit_BCE':float(loss_bits.item()),
   'presence_BCE':float(loss_presence.item()),'codeword_CE_per_bit':float(codeword.item()),'parent_bit_retention':float(retain.item()),'parent_presence_retention':float(presence_retain.item()),'clean_losses':clean_values,'projection':projection})
  if step%100==0:
   save(root/'training_history.json',history);state('training',step=step,total=p['extra_steps'],best_additional_step=best_step,elapsed_seconds=time.perf_counter()-started)
   print('V3_TRAIN',step,p['extra_steps'],'loss',round(float(loss.item()),5),'codeword',round(float(codeword.item()),5),'elapsed',round(time.perf_counter()-started,1),flush=True)
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
   print('V3_SELECTION',json.dumps({'extra_step':step,'primary':measured[11]['correct_by_condition'],'bad11':measured[11]['bad'],
    'secondary':measured[16]['correct_by_condition'],'bad16':measured[16]['bad'],'eligible':bool(rank[0]),'best_extra_step':best_step}),flush=True)
 for n in frozen+buffers:assert torch.equal(model.state_dict()[n].cpu(),initial[n]),n
 save(root/'selection_locked.json',{'complete':True,'additional_steps_completed':p['extra_steps'],'selected_additional_step':best_step,
  'selected_weight_sha256':sha(root/'selected_multitask.pth'),'parent_weight_sha256':p['parent_weight_sha256'],
  'rank':list(best_rank),'test_and_calibration_not_used_for_selection':True,'frozen_parameter_count':len(frozen),'batchnorm_buffers_unchanged':True})
 print('V3_TRAINING_COMPLETE',best_step,flush=True)

 # Selection is now locked; held development calibration is a check of the SAME gates, not a new fit.
 state('fixed_policy_calibration_check',step=p['extra_steps']);model.load_state_dict(torch.load(root/'selected_multitask.pth',map_location='cpu'),strict=True)
 raw=source.views_infer(model,records,mmap,None,'calibration');ds=source.decisions(records,raw,bch,True);calibration={}
 for k,d in ds.items():
  measured=source.metrics(d,d['bit_gate']&(d['presence']>=thresholds[k]));reference=policy['per_code'][str(k)]['new_joint']
  retention={c:measured['correct_by_condition'][c]/max(1,reference['correct_by_condition'][c]) for c in CONDITIONS}
  calibration[k]={'fixed_threshold':thresholds[k],'metrics':measured,'parent':reference,'retention':retention,
   'passed':measured['bad']==0 and min(retention.values())>=1.}
 source.save_decisions(root,'calibration_fixed',ds,thresholds);save(root/'calibration_check.json',calibration)

 # Previously observed 4096-source cohort: explicitly a regression benchmark, not independent evidence.
 state('regression_evaluation',step=p['extra_steps']);testrows=json.loads((parent/'future_test_image_manifest.json').read_text())
 testrecords=[dict(r,item_id=f"{r['group']}:{r['sample_id']}",role='test') for r in testrows]
 testcache=np.memmap(Path(p['parent_local'])/'future_pixels.uint8',mode='r',dtype=np.uint8,shape=(4096,3,512,512,3))
 assert len(testrows)==4096
 for r in testrows:assert sha(r['path'])==r['sha256']
 raw=source.views_infer(model,testrecords,testcache,None,'test');tests=source.decisions(testrecords,raw,bch,True)
 source.save_decisions(root,'regression_v3',tests,thresholds)
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
  summary.append({'k':k,'parent_v1':baseline,'extension_v3':measured,'retention':retention,'independent_test':False})
  for label,m in [('parent_v1',baseline),('extension_v3',measured)]:
   for row in m['cells']:allrows.append({'version':label,'k':k,**row})
 save(root/'regression_summary.json',summary);save(root/'paired_correct_changes.json',paired_results)
 with (root/'regression_cells.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)

 # Independently re-decode saved logits; replay real image views to catch export/order errors.
 checked=0
 for k in [11,16]:
  with np.load(root/f'regression_v3_k{k}.npz') as z:v={n:z[n].copy() for n in z.files}
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
 promising=bool(best_step>0 and all(v['passed'] for v in calibration.values()) and primary['extension_v3']['bad']==0
  and min(primary['retention'].values())>=1. and primary['extension_v3']['correct']>=primary['parent_v1']['correct']
  and primary['extension_v3']['correct_by_condition']['jpeg_q90_420']>primary['parent_v1']['correct_by_condition']['jpeg_q90_420']
  and secondary['extension_v3']['bad']<=secondary['parent_v1']['bad'] and min(secondary['retention'].values())>=1.)
 save(root/'validation.json',{'complete':True,'decisions_recomputed':checked,'source_images':4096,'image_views':12288,'real_views_replayed':replay,
  'maximum_replay_logit_difference':maximum,'fixed_gates_unchanged':True,'no_independent_test_claim':True,'regression_promising':promising})
 for path,value in p['frozen_inputs'].items():assert sha(path)==value,path
 assert sha(parent/'selected_multitask.pth')==p['parent_weight_sha256']
 save(root/'decision.json',{'version':'v3_multiview_guard','complete':True,'selected_additional_step':best_step,
  'extra_steps_completed':p['extra_steps'],'regression_promising':promising,'not_automatically_promoted':True,
  'requires_fresh_independent_cohort_before_generalization_claim':True,'parent_release_unchanged':True})
 title='V3：三视图训练与PNG/JPEG444梯度保留'
 table='<table><tr><th>版本</th><th>码</th><th>PNG正确/1024</th><th>JPEG444正确/1024</th><th>JPEG420正确/1024</th><th>错误归属总数</th></tr>'
 for entry in summary:
  for label in ['parent_v1','extension_v3']:
   m=entry[label];cells=[label,entry['k']]+[m['correct_by_condition'][c] for c in CONDITIONS]+[m['bad']]
   table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>'
 table+='</table>'
 verdict='通过预先固定的回归候选要求，保存为独立V3候选；V1发布包保持原样。' if promising else '未通过全部预先固定的回归候选要求；V1仍是保留发布版本，V3结果仅作为实验记录。'
 (root/'RESULTS_zh.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px Arial;max-width:1150px;margin:28px auto;color:#18364a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #abc;padding:10px}p{line-height:1.6}</style><h1>'+title+'</h1><p>'+verdict+'</p>'+table+
  '<p>本轮追加1600步，所选额外步数 '+str(best_step)+'。固定baseline-3000生成器、软件环境、BCH和拒绝门槛；只在原fit角色更新权重，selection选点，calibration作固定规则核验。4096源图是已观察过的回归队列，不能视作新独立测试。错误归属包含正样本认错用户与无水印图被接受；同图压缩变体不是独立样本。</p></html>')
 (root/'README_zh.md').write_text('# '+title+'\n\n'+verdict+'\n\n从V1初始化，只训练layer4与bit head，前层、BN统计及presence head全部冻结。每步同一12张fit图像同时使用PNG、JPEG444、JPEG420三个视图；V1作为PNG/444教师，加logit与正确bit margin保留。V2只在fit图像上提供正确解码且GT BCE更低的420教师输出。梯度投影使SGD更新与两份干净视图训练目标一阶不冲突；这不保证未见图像不退化，仍以开发与回归结果核验。\n\nSGD无momentum、无weight decay、统一LR0.002，100步warmup和cosine到10%，梯度范数裁剪1，共1600步，每400步选点，起点也作为备选。选点要求两码无错误归属、PNG/444正确数均不低于V1、420不低于V1，再按k11 JPEG420正确数、总正确数、k16同类计数排序。calibration与回归也要求每条件不退化，主码420严格改善，拒绝门槛不调整。\n\n旧开发和4096源图回归队列均已在历史研究观察，不能称为全新独立测试。所有fit/selection/calibration角色保持隔离。V1与V2仅作为只读参考；本目录是单独V3版本，不自动替换发布包。权重、GT、logits、概率、决策、优化器/RNG、训练和选点记录均保留。\n')
 portable_config=json.loads((Path(p['portable_base_repo'])/'configs/pretrained.json').read_text())
 portable_config.update({'weights':'selected_multitask.pth','weights_relative_to':'configuration_directory','weights_sha256':sha(root/'selected_multitask.pth'),
  'version':'v3_multiview_guard','parent_version':'v1_negative_aware_2400','trained':True,'regression_promising':promising})
 save(root/'inference_config.json',portable_config)
 (root/'LOAD_V3_IN_V1.md').write_text('# 在便携推理包中显式加载V3\n\n将本候选包解压到V1便携仓库中的 experiments/v3/，不替换 models/ 下的V1权重或 configs/pretrained.json。然后在V1仓库根目录运行：\n\n```bash\npython scripts/run_locked.py infer.py --config experiments/v3/inference_config.json --image YOUR_512_IMAGE.png --code 11 --output outputs/v3_prediction.json\n```\n\ninference_config.json引用同目录的selected_multitask.pth并校验完整SHA256；BCH及环境配置使用原V1根目录的同一套冻结文件。默认不带--config仍加载V1。归属结果需同时保留所用配置和权重hash以区分版本。此包是独立研究候选，尚未自动替换V1发布版；regression_promising不等于全新独立评估通过。\n')
 files=[x for x in sorted(root.iterdir()) if x.is_file() and x.name not in ['pipeline.log','status.json','SHA256SUMS.txt','CANDIDATE_BUNDLE.zip','BUNDLE_SHA256SUMS.txt']]
 (root/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(x)}  {x.name}' for x in files)+'\n')
 for line in (root/'SHA256SUMS.txt').read_text().splitlines():
  expected,name=line.split('  ',1);assert sha(root/name)==expected,name
 # A compact self-describing trained candidate bundle is separate from the unchanged V1 GitHub ZIP.
 bundle_names=['selected_multitask.pth','inference_config.json','LOAD_V3_IN_V1.md','fixed_policy.json','plan.json','protocol_lock.json','selection_locked.json','decision.json','regression_summary.json',
   'regression_cells.csv','RESULTS_zh.html','README_zh.md','validation.json','wouaf_extension_v3.py','frozen_parent_recipe.py','preflight.json']
 (root/'BUNDLE_SHA256SUMS.txt').write_text('\n'.join(f'{sha(root/name)}  {name}' for name in bundle_names)+'\n')
 with zipfile.ZipFile(local/'CANDIDATE_BUNDLE.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for name in bundle_names+['BUNDLE_SHA256SUMS.txt']:
   z.write(root/name,arcname=name)
 shutil.copy2(local/'CANDIDATE_BUNDLE.zip',root/'CANDIDATE_BUNDLE.zip')
 with zipfile.ZipFile(root/'CANDIDATE_BUNDLE.zip') as z:
  assert z.testzip() is None and hashlib.sha256(z.read('selected_multitask.pth')).hexdigest()==sha(root/'selected_multitask.pth')
 state('complete',step=p['extra_steps'],best_additional_step=best_step,regression_promising=promising,bundle_sha256=sha(root/'CANDIDATE_BUNDLE.zip'))
 save(root.parent.parent/'LATEST_V3_TRAINING.json',{'root':str(root),'version':'v3_multiview_guard','state':'complete',
  'regression_promising':promising,'selected_additional_step':best_step,'parent_release_unchanged':True})
 print('V3_COMPLETE',json.dumps({'root':str(root),'best_extra_step':best_step,'regression_promising':promising,'summary':summary}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 try:run(args.root,args.resume)
 except Exception:
  save(args.root/'status.json',{'stage':'failed','error':traceback.format_exc(),'updated_unix':time.time()});raise

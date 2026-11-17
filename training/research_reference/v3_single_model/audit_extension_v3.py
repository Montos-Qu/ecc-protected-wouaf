"""Independent saved-artifact audit for a completed V3 experiment."""
import argparse,csv,hashlib,json,math,sys,zipfile
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
plan=json.loads((root/'plan.json').read_text());assert sha(root/'plan.json')==json.loads((root/'protocol_lock.json').read_text())['plan_sha256']
assert sha(root/'wouaf_extension_v3.py')==plan['script_sha256']
status=json.loads((root/'status.json').read_text());assert status['stage']=='complete'
parent=Path(plan['parent_root']);base=Path(plan['portable_base_repo']);sys.path.insert(0,str(base))
from ecc_wouaf.core import Decoder,load_config,reference_runtime
reference_runtime();decoder=Decoder(load_config())
records=json.loads((parent/'development_image_manifest.json').read_text());history=json.loads((root/'training_history.json').read_text())
assert len(history)==1600
for step,row in enumerate(history,1):
 assert row['additional_step']==step and row['condition']=='paired_three_views'
 ids=row['fit_indices'];assert len(ids)==12 and all(records[i]['role']=='fit' for i in ids)
 assert [records[i]['group'] for i in ids[:4]]==['positive_k11']*2+['positive_k16']*2
 assert all(records[i]['group']=='generated_negative' for i in ids[4:10])
 assert all(records[i]['group']=='natural_negative' for i in ids[10:])
 assert [records[i]['sample_id'] for i in ids[:4]]==[records[i]['sample_id'] for i in ids[4:8]]
 assert all(math.isfinite(row[n]) for n in ['loss','bit_BCE','presence_BCE','codeword_CE_per_bit','parent_bit_retention','parent_presence_retention'])
original=torch.load(parent/'selected_multitask.pth',map_location='cpu')
last=torch.load(root/'last_resumable_checkpoint.pth',map_location='cpu');selected=torch.load(root/'selected_multitask.pth',map_location='cpu')
assert last['step']==1600 and last['best_weight_sha256']==sha(root/'selected_multitask.pth')
frozen=[name for name in original if not name.startswith(('backbone.layer4.','bit_head.'))
 or any(k in name for k in ['running_mean','running_var','num_batches_tracked'])]
for name in frozen:assert torch.equal(original[name],last['state_dict'][name]) and torch.equal(original[name],selected[name]),name
assert len(last['optimizer']['param_groups'])==1
group=last['optimizer']['param_groups'][0]
assert group['initial_rate']==plan['learning_rate'] and group['momentum']==group['weight_decay']==0
assert abs(group['lr']/group['initial_rate']-.1)<1e-10 and last['optimizer']['state']=={}
for row in history:
 assert len(row['clean_losses'])==2 and all(math.isfinite(x) for x in row['clean_losses'])
 assert min(row['projection']['after'])>=-1e-6
 assert all(math.isfinite(x) for key in ['before','after','multipliers'] for x in row['projection'][key])
table={(r['version'],int(r['k']),r['condition'],r['group']):r for r in csv.DictReader((root/'regression_cells.csv').open(encoding='utf-8-sig'))}
assert len(table)==36
decisions=0;cells=0;total_correct={}
for label in ['parent_v1','extension_v3']:
 for k in [11,16]:
  path=parent/f'future_test_negative_aware_k{k}.npz' if label=='parent_v1' else root/f'regression_v3_k{k}.npz'
  with np.load(path) as z:d={name:z[name].copy() for name in z.files}
  decoded=decoder.decode(d['logits'],d['presence'],k)
  for new,old in [('accepted','final_accept'),('candidate_user_id','candidate_user'),('ecc_accept','ecc'),('pred_bits','pred_bits'),('distance32','distance32')]:
   assert np.array_equal(decoded[new],d[old]),(label,k,new)
  assert np.allclose(decoded['probabilities'],d['probabilities'],atol=0,rtol=0)
  truth=(d['group']==f'positive_k{k}')&decoded['ecc_accept']&(decoded['candidate_user_id']==d['gt_user_id'])
  total_correct[(label,k)]=int((truth&decoded['accepted']).sum())
  for condition in ['png8_control','jpeg_q90_444','jpeg_q90_420']:
   for group in [f'positive_k{k}','generated_negative','natural_negative']:
    mask=(d['condition']==condition)&(d['group']==group);row=table[(label,k,condition,group)]
    actual={'images':int(mask.sum()),'correct':int((mask&decoded['accepted']&truth).sum()),
     'bad':int((mask&decoded['accepted']&~truth).sum()),'reject':int((mask&~decoded['accepted']).sum())}
    assert all(int(row[n])==v for n,v in actual.items());cells+=1
  valid=d['gt_bits_valid'];assert np.array_equal(d['bit_errors'],np.where(valid,(d['gt_bits']!=d['pred_bits']).sum(axis=1),-1))
  decisions+=len(d['logits'])
assert decisions==36864 and cells==36
for row in json.loads((root/'paired_correct_changes.json').read_text()):
 k=row['k'];assert row['new_correct_only']-row['parent_correct_only']==row['net_correct']==total_correct[('extension_v3',k)]-total_correct[('parent_v1',k)]
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(root/name)==expected,name
with zipfile.ZipFile(root/'CANDIDATE_BUNDLE.zip') as z:
 assert z.testzip() is None
 for line in z.read('BUNDLE_SHA256SUMS.txt').decode().splitlines():
  expected,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
for path,value in plan['frozen_inputs'].items():assert sha(path)==value,path
result={'complete':True,'all_1600_fit_batches_audited':True,'all_losses_finite':True,'paired_content_sampling_verified':True,
 'frozen_parameters_and_BN_state_entries':len(frozen),'selected_and_terminal_frozen_entries_unchanged':True,
 'training_update_count':1600,'optimizer':'stateless_SGD','recorded_two_clean_gradient_constraints_verified':True,'optimizer_final_learning_rates_verified':True,'independently_recomputed_decisions':decisions,
 'result_table_cells_verified':cells,'gt_bits_probabilities_and_hamming_verified':True,'full_bundle_sha256_entries_verified':True,
 'parent_release_and_all_frozen_inputs_unchanged':True,'independent_test':False,'selected_model_sha256':sha(root/'selected_multitask.pth'),
 'parent_model_sha256':plan['parent_weight_sha256'],'bundle_sha256':sha(root/'CANDIDATE_BUNDLE.zip')}
save(root/'independent_audit.json',result)
print('V3_INDEPENDENT_AUDIT_PASSED',json.dumps(result),flush=True)

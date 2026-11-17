"""Independent delivery checks; no fitting, threshold selection or model updates."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(root,local):
    import torch
    p=json.loads((root/'plan.json').read_text())
    assert sha(root/'plan.json')==json.loads((root/'protocol_lock.json').read_text())['plan_sha256']
    assert sha(root/'wouaf_negative_aware.py')==p['script_sha256']
    for path,value in p['frozen_inputs'].items():assert sha(path)==value,path
    for name,value in json.loads((root/'data_lock.json').read_text()).items():assert sha(root/name)==value,name
    assert json.loads((root/'test_inference_verification.json').read_text())['complete']
    development=json.loads((root/'development_image_manifest.json').read_text())
    history=json.loads((root/'training_history.json').read_text())
    assert len(history)==2400 and [r['step'] for r in history]==list(range(1,2401))
    for r in history:
        ix=r['fit_indices'];assert len(ix)==24 and all(development[i]['role']=='fit' for i in ix)
        assert [development[i]['group'] for i in ix[:8]]==['positive_k11']*4+['positive_k16']*4
        assert [development[i]['group'] for i in ix[8:20]]==['generated_negative']*12
        assert [development[i]['group'] for i in ix[20:]]==['natural_negative']*4
        assert [development[i]['sample_id'] for i in ix[:8]]==[development[i]['sample_id'] for i in ix[8:16]]
        assert r['condition']==['png8_control','jpeg_q90_444','jpeg_q90_420','jpeg_q90_420'][(r['step']-1)%4]
    role_by_content={};role_by_user={}
    for r in development:
        role_by_content.setdefault(r['content_key'],set()).add(r['role'])
        if r['user_id']>=0:role_by_user.setdefault((r['k'],r['user_id']),set()).add(r['role'])
    assert all(len(v)==1 for v in list(role_by_content.values())+list(role_by_user.values()))
    selection=json.loads((root/'selection_history.json').read_text())
    assert [r['step'] for r in selection]==[400,800,1200,1600,2000,2400]
    winner=max(selection,key=lambda r:tuple(r['rank']))
    lock=json.loads((root/'selection_locked.json').read_text())
    assert winner['step']==lock['selected_step'] and sha(root/'selected_multitask.pth')==lock['selected_weights_sha256']
    original=torch.load(p['initial_extractor'],map_location='cpu');selected=torch.load(root/'selected_multitask.pth',map_location='cpu')
    frozen_tensors=0
    for name,value in original.items():
        if name.startswith('fc.') or (name.startswith(('layer3.','layer4.')) and not any(t in name for t in ['running_mean','running_var','num_batches_tracked'])):continue
        assert torch.equal(selected['backbone.'+name],value),name;frozen_tensors+=1
    policy=json.loads((root/'locked_presence_policy.json').read_text());policy_hash=sha(root/'locked_presence_policy.json')
    bit_policies=json.loads(Path(p['old_policy']).read_text())
    audited=0
    for k in [11,16]:
        with np.load(root/f'calibration_k{k}.npz',allow_pickle=False) as z:cal={n:z[n].copy() for n in z.files}
        bad=cal['bit_gate']&cal['bad_candidate']
        expected=max(0.,float(np.nextafter(cal['presence'][bad].max(),np.inf))) if bad.any() else 0.
        assert expected==policy['per_code'][str(k)]['threshold']
        assert np.array_equal(cal['final_accept'],cal['bit_gate']&(cal['presence']>=expected))
        assert not (cal['final_accept']&cal['bad_candidate']).any()
        with np.load(root/f'future_test_inputs_k{k}.npz') as z:
            users=set(z['user_id'].tolist());assert len(users)==1024
            assert not users&{r['user_id'] for r in development if r['k']==k}
            assert np.array_equal(z['sample_id'],np.arange(2420,3444))
        for model in ['old200','candidate300','negative_aware']:
            with np.load(root/f'future_test_{model}_k{k}.npz',allow_pickle=False) as z:d={n:z[n].copy() for n in z.files}
            assert len(d['logits'])==9216
            assert np.isfinite(d['logits']).all() and np.isfinite(d['presence']).all()
            assert np.array_equal(d['probabilities'],torch.from_numpy(d['logits']).sigmoid().numpy())
            assert np.array_equal(d['pred_bits'],d['probabilities']>.5)
            assert np.array_equal(d['gt_bits_valid'],d['positive'])
            assert np.array_equal(d['bit_gate'],d['ecc']&(d['candidate_loglikelihood']>=bit_policies['policies']['strict_zero_observed'][str(k)]['threshold']))
            accepted=d['bit_gate'] if model!='negative_aware' else d['bit_gate']&(d['presence']>=expected)
            assert np.array_equal(accepted,d['final_accept'])
            assert not (d['final_accept']&~d['bit_gate']).any()
            assert np.array_equal(d['correct_candidate'],d['positive']&d['ecc']&(d['candidate_user']==d['gt_user_id']))
            assert np.array_equal(d['bit_errors'],np.where(d['positive'],(d['pred_bits']!=d['gt_bits']).sum(axis=1),-1))
            audited+=len(accepted)
    with (root/'test_cell_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==54
    for row in rows:
        with np.load(root/f"future_test_{row['model']}_k{row['k']}.npz") as z:
            mask=(z['group']==row['group'])&(z['condition']==row['condition']);a=z['final_accept']
            observed=[int(mask.sum()),int((mask&a&z['correct_candidate']).sum()),int((mask&a&z['bad_candidate']).sum()),int((mask&~a).sum())]
            assert observed==[int(row[n]) for n in ['images','correct','bad','reject']]
    # Replay selected network outputs from archived source pixels, without gradients.
    spec=importlib.util.spec_from_file_location('locked_negative_aware_model',root/'wouaf_negative_aware.py')
    implementation=importlib.util.module_from_spec(spec);spec.loader.exec_module(implementation)
    implementation.configure(False);network=implementation.make_model(p);network.load_state_dict(selected,strict=True);network.eval()
    bch=implementation.BCH(p);replay=0
    for k,groups in [(11,['positive_k11','generated_negative','natural_negative']),(16,['positive_k16'])]:
        with np.load(root/f'future_test_negative_aware_k{k}.npz') as z:d={n:z[n].copy() for n in z.files}
        for group in groups:
            for condition in implementation.CONDITIONS:
                ix=np.flatnonzero((d['group']==group)&(d['condition']==condition))[:8]
                files=[local/'future_pngs'/group/f"{int(i)}.png" for i in d['sample_id'][ix]]
                batch=np.stack([implementation.pixels(path,condition) for path in files])
                with torch.no_grad():bits,presence=network(implementation.normalized(torch,batch))
                values=bits.cpu().numpy();scores=presence.cpu().numpy()
                assert np.allclose(values,d['logits'][ix],atol=1e-4,rtol=1e-4)
                assert np.allclose(scores,d['presence'][ix],atol=1e-4,rtol=1e-4)
                assert np.array_equal(torch.from_numpy(values).sigmoid().numpy()>.5,d['pred_bits'][ix])
                decision=bch.decide(values,k,True)
                assert np.array_equal(decision['bit_gate']&(scores>=policy['per_code'][str(k)]['threshold']),d['final_accept'][ix])
                replay+=len(ix)
    assert replay==96 and audited==55296 and sha(root/'locked_presence_policy.json')==policy_hash
    source=json.loads((root/'future_test_generation.json').read_text())
    assert sha(root/'future_test_source_images.zip')==source['archived_sha256']
    image_rows=json.loads((root/'future_test_image_manifest.json').read_text())
    assert len(image_rows)==4096
    for r in image_rows:assert sha(r['path'])==r['sha256']
    photos=[r for r in image_rows if r['group']=='natural_negative']
    previous=json.loads((root/'future_test_photo_selection.json').read_text())
    assert not {r['sample_id'] for r in photos}&set(previous['prior_excluded'])
    assert not {r['original_sha256'] for r in photos}&set(previous['prior_photo_hashes'])
    verification={'complete':True,'all2400_training_batches_fit_only_and_content_paired':True,'saved_checkpoint_chosen_only_by_predeclared_selection_rank':True,
                  'original_frozen_parameter_and_buffer_tensors_verified':frozen_tensors,'calibration_thresholds_independently_recomputed':2,
                  'future_test_decisions_independently_checked':audited,'all54CSVcells_recomputed':True,'selected_network_replayed_image_views':96,
                  'source_archive_and_4096_image_hashes_verified':True,'no_test_based_retuning_or_active_promotion':True}
    (root/'independent_delivery_audit.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('INDEPENDENT_NEGATIVE_AWARE_AUDIT_PASSED',json.dumps(verification),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--local',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.local)

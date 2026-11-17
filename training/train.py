"""Explicit opt-in training; published weights are never overwritten."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from ecc_wouaf.core import reference_runtime,CONDITIONS,Decoder,load_config,normalize_pixels,image_pixels,sha256
from recipe import GROUPS,dataset,initialize,image_cache,infer,decision_data,metrics,rule,save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--init',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();reference_runtime();torch.manual_seed(20261029)
    assert a.steps>100 and not a.out.resolve().is_relative_to((ROOT/'models').resolve())
    records,targets=dataset(a.data);model,initial=initialize(a.init)
    config=load_config();decoder=Decoder(config)
    pools={g:np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']==g]) for g in GROUPS}
    paired={r['sample_id']:i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='generated_negative'}
    assert all(records[i]['sample_id'] in paired for g in GROUPS[:2] for i in pools[g])
    if a.dry_run:
        # Inference-only input/initialization check: no optimizer is constructed or stepped.
        ix=np.array([i for i,r in enumerate(records) if r['group']=='positive_k11'][:4])
        with torch.no_grad():
            for c,condition in enumerate(CONDITIONS):
                batch=np.stack([image_pixels(a.data/records[i]['image'],condition) for i in ix])
                values,_=model(normalize_pixels(batch,'cuda'));values=values.cpu().numpy()
                assert np.allclose(values,targets['teacher'][ix,c],atol=1e-4,rtol=1e-4)
        print('TRAINING_DRY_RUN_PASSED',len(records),'images;12 reference logits; zero optimizer steps');return
    a.out=a.out.resolve()
    if not a.resume:a.out.mkdir(parents=True,exist_ok=False)
    else:assert (a.out/'last_checkpoint.pth').is_file()
    run_plan={'steps':a.steps,'seed':20261029,'initial_sha256':sha256(a.init),'data_manifest_sha256':sha256(a.data/'manifest.json'),
              'training_code_sha256':sha256(Path(__file__)),'core_code_sha256':sha256(ROOT/'ecc_wouaf/core.py'),
              'fit_roles':['fit'],'selection_role':'selection','calibration_not_used_for_selection':True,
              'loss_weights':{'bit':2,'presence':1,'negative_uncertainty':.05,'teacher_MSE':.05,'paired_margin':.2}}
    if a.resume:assert json.loads((a.out/'plan.json').read_text())==run_plan
    else:save(a.out/'plan.json',run_plan)
    cache=image_cache(a.data,records,a.out/'pixels.uint8')
    selection_ix=np.array([i for i,r in enumerate(records) if r['role']=='selection'])
    ref={'index':np.tile(selection_ix,3),'condition':np.repeat(CONDITIONS,len(selection_ix)),
         'logits':np.concatenate([targets['teacher'][selection_ix,c] for c in range(3)]),'presence':np.zeros(3*len(selection_ix),np.float32)}
    ref=decision_data(records,ref,decoder);baseline={k:metrics(d,d['bit_gate']) for k,d in ref.items()}
    groups=[{'params':[v for n,v in model.named_parameters() if n.startswith(prefix) and v.requires_grad],'lr':lr,'initial_rate':lr}
            for prefix,lr in [('backbone.layer3.',2e-6),('backbone.layer4.',5e-6),('bit_head.',1e-5),('presence_head.',1e-4)]]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);random=np.random.default_rng(20261029)
    history=[];assessments=[];best_rank=None;best_step=None;start=1
    if a.resume:
        state=torch.load(a.out/'last_checkpoint.pth');model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
        random.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['step']+1;best_step=state['best_step'];best_rank=tuple(state['best_rank'])
        history=json.loads((a.out/'training_history.json').read_text());assessments=json.loads((a.out/'selection_history.json').read_text())
    trainable=[v for v in model.parameters() if v.requires_grad];bce=nn.BCEWithLogitsLoss()
    for step in range(start,a.steps+1):
        positive=np.concatenate([random.choice(pools[g],4,replace=False) for g in GROUPS[:2]])
        matched=np.array([paired[records[i]['sample_id']] for i in positive])
        ix=np.concatenate([positive,matched,random.choice(pools['generated_negative'],4,replace=False),random.choice(pools['natural_negative'],4,replace=False)])
        c=[0,1,2,2][(step-1)%4];assert all(records[i]['role']=='fit' for i in ix)
        model.eval();model.presence_head.train();optimizer.zero_grad(set_to_none=True)
        bits,presence=model(normalize_pixels(cache[ix,c],'cuda'))
        gt=torch.from_numpy(targets['gt_bits'][positive].copy()).float().cuda();teacher=torch.from_numpy(targets['teacher'][positive,c].copy()).cuda()
        bit_loss=bce(bits[:8],gt)
        presence_loss=.5*bce(presence[:8],torch.ones_like(presence[:8]))+.25*bce(presence[8:20],torch.zeros_like(presence[8:20]))+.25*bce(presence[20:],torch.zeros_like(presence[20:]))
        neutral=.5*bce(bits[8:20],torch.full_like(bits[8:20],.5))+.5*bce(bits[20:],torch.full_like(bits[20:],.5))
        retention=(bits[:8]-teacher).square().mean();paired_loss=torch.nn.functional.softplus(1-presence[:8]+presence[8:16]).mean()
        loss=2*bit_loss+presence_loss+.05*neutral+.05*retention+.2*paired_loss
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(trainable,1.)
        factor=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0.,(step-100)/(a.steps-100)))))
        for g in optimizer.param_groups:g['lr']=g['initial_rate']*factor
        optimizer.step();history.append({'step':step,'fit_indices':ix.tolist(),'condition':CONDITIONS[c],'loss':float(loss.item())})
        if step%100==0:save(a.out/'training_history.json',history);print('TRAIN',step,a.steps,float(loss.item()),flush=True)
        if step%400==0 or step==a.steps:
            ds=decision_data(records,infer(model,records,cache,'selection'),decoder);measured={};thresholds={};retention={}
            for k,d in ds.items():
                thresholds[k],measured[k]=rule(d)
                retention[k]={c:measured[k]['correct_by_condition'][c]/max(1,baseline[k]['correct_by_condition'][c]) for c in CONDITIONS}
            primary=min(retention[11].values());rank=(primary>=.95,primary,measured[11]['correct'],min(retention[16].values()),measured[16]['correct'])
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            torch.save(state,a.out/f'checkpoint_{step:04d}.pth')
            if best_rank is None or rank>best_rank:best_rank=rank;best_step=step;torch.save(state,a.out/'selected.pth')
            assessments.append({'step':step,'rank':list(rank),'selection_metrics':measured,'retention':retention,'selection_thresholds':thresholds})
            save(a.out/'selection_history.json',assessments);save(a.out/'training_history.json',history)
            torch.save({'state_dict':state,'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':random.bit_generator.state,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best_rank':list(best_rank),'best_step':best_step},a.out/'last_checkpoint.pth')
    for name,value in initial.items():
        if name.startswith('fc.') or (name.startswith(('layer3.','layer4.')) and not any(t in name for t in ['running_mean','running_var','num_batches_tracked'])):continue
        assert torch.equal(model.state_dict()['backbone.'+name].cpu(),value),name
    save(a.out/'selection_locked.json',{'complete':True,'selected_step':best_step,'weights_sha256':sha256(a.out/'selected.pth'),
                                     'calibration_not_observed':True,'evaluation_not_observed':True})
    print('TRAINING_COMPLETE',best_step)
if __name__=='__main__':main()

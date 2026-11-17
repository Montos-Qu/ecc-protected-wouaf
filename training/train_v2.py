"""Explicit V2 continuation of V1; fixed gates and full BCH codeword objective."""
import argparse,json,math,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ecc_wouaf.core import Extractor,Decoder,load_config,reference_runtime,normalize_pixels,image_pixels,sha256,CONDITIONS
from recipe import GROUPS,dataset,image_cache,infer,decision_data,metrics,save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'models/v1_negative_aware_2400.pth')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    reference_runtime();torch.manual_seed(20261030);records,targets=dataset(a.data)
    assert a.steps>100 and not a.out.resolve().is_relative_to((ROOT/'models').resolve())
    model=Extractor();model.load_state_dict(torch.load(a.parent,map_location='cpu'),strict=True)
    for name,v in model.named_parameters():v.requires_grad_(name.startswith(('backbone.layer3.','backbone.layer4.','bit_head.','presence_head.')))
    model.eval().float().cuda();initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    frozen=[n for n,v in model.named_parameters() if not v.requires_grad]+[n for n,_ in model.named_buffers()]
    config=load_config();decoder=Decoder(config);books={}
    for k in [11,16]:
        users=np.arange(2**k,dtype=np.uint32);message=((users[:,None]>>np.arange(k-1,-1,-1))&1).astype(np.uint8)
        word=message@decoder.books[k][2]%2;full=np.column_stack([word,word.sum(axis=1)%2]).astype(np.uint8)
        for i,r in enumerate(records):
            if r['group']==f'positive_k{k}':assert np.array_equal(full[r['user_id']],targets['gt_bits'][i])
        books[k]=torch.from_numpy(full.astype(np.float32)).cuda()
    if a.dry_run:
        assert sha256(a.parent)==json.loads((ROOT/'configs/versions/v1.json').read_text())['weights_sha256'],'Reference dry-run uses published V1.'
        with np.load(ROOT/'training/reference/v1_fit_teacher.npz') as z:teacher={n:z[n].copy() for n in z.files}
        assert np.array_equal(teacher['item_id'],[r['item_id'] for r in records])
        ix=np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='positive_k11'][:4])
        with torch.no_grad():
            for c,condition in enumerate(CONDITIONS):
                batch=np.stack([image_pixels(a.data/records[i]['image'],condition) for i in ix])
                bits,pres=model(normalize_pixels(batch,'cuda'));bits=bits.cpu().numpy();pres=pres.cpu().numpy()
                assert np.allclose(bits,teacher['logits'][ix,c],atol=1e-4,rtol=1e-4)
                assert np.allclose(pres,teacher['presence'][ix,c],atol=1e-4,rtol=1e-4)
        scores=torch.from_numpy(bits).cuda()@books[11].T;users=torch.tensor([records[i]['user_id'] for i in ix],device='cuda')
        direct=(torch.logsumexp(scores,1)-scores[torch.arange(len(ix),device='cuda'),users]).mean()
        assert torch.allclose(F.cross_entropy(scores,users),direct,atol=1e-5,rtol=1e-5)
        print('V2_DRY_RUN_PASSED',len(records),'images;12 parent views;2048/65536 codebooks;zero optimizer steps');return
    if a.resume:assert (a.out/'last_checkpoint.pth').exists()
    else:a.out.mkdir(parents=True,exist_ok=False)
    plan={'steps':a.steps,'seed':20261030,'parent_sha256':sha256(a.parent),'manifest_sha256':sha256(a.data/'manifest.json'),
        'code_sha256':sha256(Path(__file__)),'fixed_policies':config['policies'],'condition_schedule':[0,2,1,2,2]}
    if a.resume:assert json.loads((a.out/'plan.json').read_text())==plan
    else:save(a.out/'plan.json',plan)
    cache=image_cache(a.data,records,a.out/'pixels.uint8')
    # The teacher is generated only from the current parent and only on fit roles.
    with torch.no_grad():raw=infer(model,records,cache,'fit')
    teacher_bits=np.zeros((len(records),3,32),np.float32);teacher_presence=np.zeros((len(records),3),np.float32)
    valid=np.zeros((len(records),3),bool)
    for c,condition in enumerate(CONDITIONS):
        chosen=raw['condition']==condition;ix=raw['index'][chosen]
        teacher_bits[ix,c]=raw['logits'][chosen];teacher_presence[ix,c]=raw['presence'][chosen];valid[ix,c]=True
    ds=decision_data(records,infer(model,records,cache,'selection'),decoder)
    parent_metrics={k:metrics(d,d['accepted']) for k,d in ds.items()}
    def rank(values):
        ratios={k:{c:values[k]['correct_by_condition'][c]/max(1,parent_metrics[k]['correct_by_condition'][c]) for c in CONDITIONS} for k in [11,16]}
        eligible=all(values[k]['bad']==0 for k in [11,16]) and min(ratios[11].values())>=.98 and min(ratios[16].values())>=.95
        return (eligible,values[11]['correct_by_condition']['jpeg_q90_420'],values[11]['correct'],values[16]['correct_by_condition']['jpeg_q90_420'],values[16]['correct']),ratios
    pools={g:np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']==g]) for g in GROUPS}
    paired={r['sample_id']:i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='generated_negative'}
    groups=[{'params':[v for n,v in model.named_parameters() if n.startswith(prefix) and v.requires_grad],'lr':rate,'initial_rate':rate}
        for prefix,rate in [('backbone.layer3.',1e-6),('backbone.layer4.',2.5e-6),('bit_head.',5e-6),('presence_head.',2e-5)]]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);rng=np.random.default_rng(20261030)
    history=[];selection=[];best_rank,_=rank(parent_metrics);best_step=0;start=1
    if a.resume:
        state=torch.load(a.out/'last_checkpoint.pth',map_location='cpu');model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        best_step=state['best_step'];best_rank=tuple(state['best_rank']);start=state['step']+1
        history=json.loads((a.out/'training_history.json').read_text());selection=json.loads((a.out/'selection_history.json').read_text())
    else:shutil.copy2(a.parent,a.out/'selected.pth')
    trainable=[v for v in model.parameters() if v.requires_grad];bce=nn.BCEWithLogitsLoss()
    for step in range(start,a.steps+1):
        positive=np.concatenate([rng.choice(pools[g],4,replace=False) for g in GROUPS[:2]])
        matched=np.array([paired[records[i]['sample_id']] for i in positive])
        ix=np.concatenate([positive,matched,rng.choice(pools['generated_negative'],4,replace=False),rng.choice(pools['natural_negative'],4,replace=False)])
        c=[0,2,1,2,2][(step-1)%5];assert len(ix)==24 and valid[ix,c].all()
        model.eval();model.presence_head.train();optimizer.zero_grad(set_to_none=True)
        bits,presence=model(normalize_pixels(cache[ix,c],'cuda'));gt=torch.from_numpy(targets['gt_bits'][positive].copy()).float().cuda()
        bit_loss=bce(bits[:8],gt)
        presence_loss=.5*bce(presence[:8],torch.ones_like(presence[:8]))+.25*bce(presence[8:20],torch.zeros_like(presence[8:20]))+.25*bce(presence[20:],torch.zeros_like(presence[20:]))
        neutral=.5*bce(bits[8:20],torch.full_like(bits[8:20],.5))+.5*bce(bits[20:],torch.full_like(bits[20:],.5))
        retain=(bits[:8]-torch.from_numpy(teacher_bits[positive,c].copy()).cuda()).square().mean()
        old_presence=torch.from_numpy(teacher_presence[ix,c].copy()).cuda()
        presence_retain=(presence.clamp(-10,10)-old_presence.clamp(-10,10)).square().mean()
        paired_loss=F.softplus(1-presence[:8]+presence[8:16]).mean();ce=[]
        for k,begin in [(11,0),(16,4)]:
            users=torch.tensor([records[i]['user_id'] for i in positive[begin:begin+4]],device='cuda')
            ce.append(F.cross_entropy(bits[begin:begin+4]@books[k].T,users)/32)
        loss=2*bit_loss+presence_loss+.05*neutral+(.01 if c==2 else .1)*retain+.01*presence_retain+.2*paired_loss+.5*(ce[0]+ce[1])
        assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)
        factor=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0.,(step-100)/(a.steps-100)))))
        for g in optimizer.param_groups:g['lr']=g['initial_rate']*factor
        optimizer.step();history.append({'step':step,'condition':CONDITIONS[c],'fit_indices':ix.tolist(),'loss':float(loss.item())})
        if step%100==0:save(a.out/'training_history.json',history);print('V2_TRAIN',step,a.steps,float(loss.item()),flush=True)
        if step%400==0 or step==a.steps:
            ds=decision_data(records,infer(model,records,cache,'selection'),decoder);values={k:metrics(d,d['accepted']) for k,d in ds.items()}
            candidate,retention=rank(values);snapshot={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            torch.save(snapshot,a.out/f'checkpoint_{step:04d}.pth')
            if candidate>best_rank:best_rank=candidate;best_step=step;torch.save(snapshot,a.out/'selected.pth')
            selection.append({'step':step,'rank':list(candidate),'metrics':values,'retention':retention});save(a.out/'selection_history.json',selection);save(a.out/'training_history.json',history)
            torch.save({'state_dict':snapshot,'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best_step':best_step,'best_rank':list(best_rank)},a.out/'last_checkpoint.pth')
    for name in frozen:assert torch.equal(model.state_dict()[name].cpu(),initial[name]),name
    config.pop('_configuration_directory',None);config.update({'weights':'selected.pth','weights_relative_to':'configuration_directory','weights_sha256':sha256(a.out/'selected.pth'),'version':'user_v2_run'})
    save(a.out/'inference_config.json',config)
    save(a.out/'selection_locked.json',{'complete':True,'selected_extra_step':best_step,'weights_sha256':config['weights_sha256'],
        'policies_unchanged':True,'calibration_and_test_not_used_for_selection':True,'not_automatically_promoted':True})
    print('V2_COMPLETE',best_step,'extra steps selected;evaluate separately;published V3 unchanged')
if __name__=='__main__':main()

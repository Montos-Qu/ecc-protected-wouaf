"""Negative-aware multitask WOUAF extractor; frozen generator and software stack."""
import argparse
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CONDITIONS=['png8_control','jpeg_q90_444','jpeg_q90_420']
GROUPS=['positive_k11','positive_k16','generated_negative','natural_negative']
ROLES={0:'fit',1:'fit',2:'selection',3:'calibration'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def context(root):
    p=json.loads((root/'plan.json').read_text())
    assert sha(root/'plan.json')==json.loads((root/'protocol_lock.json').read_text())['plan_sha256']
    assert sha(Path(__file__))==p['script_sha256']
    for name,value in p['frozen_inputs'].items():assert sha(name)==value,name
    assert p['steps']==2400 and p['test_per_group']==1024 and p['fit_folds']==[0,1]
    return p

def pixels(path,condition):
    from PIL import Image,JpegImagePlugin
    with Image.open(path) as f:image=f.convert('RGB').copy()
    assert image.size==(512,512)
    if condition!='png8_control':
        buf=io.BytesIO();sample=0 if condition.endswith('444') else 2
        image.save(buf,format='JPEG',quality=90,subsampling=sample,optimize=False,progressive=False);buf.seek(0)
        with Image.open(buf) as f:
            assert JpegImagePlugin.get_sampling(f)==sample;image=f.convert('RGB').copy()
    return np.asarray(image).copy()

def configure(training=False):
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261029)
    torch.set_grad_enabled(training)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return torch

def make_model(p):
    import torch
    from torch import nn
    from torchvision.models import resnet50
    class Multitask(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=resnet50(weights=None);self.backbone.fc=nn.Identity()
            self.bit_head=nn.Linear(2048,32)
            self.presence_head=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,1))
        def forward(self,x):
            z=self.backbone(x);return self.bit_head(z),self.presence_head(z)[:,0]
    model=Multitask()
    original=torch.load(p['initial_extractor'],map_location='cpu')
    model.backbone.load_state_dict({k:v for k,v in original.items() if not k.startswith('fc.')},strict=True)
    model.bit_head.load_state_dict({k.removeprefix('fc.'):v for k,v in original.items() if k.startswith('fc.')},strict=True)
    for name,v in model.named_parameters():v.requires_grad_(name.startswith(('backbone.layer3.','backbone.layer4.','bit_head.','presence_head.')))
    return model.eval().float().cuda()

def normalized(torch,batch):
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    x=torch.from_numpy(np.asarray(batch).astype(np.float32)/255).permute(0,3,1,2).contiguous().cuda()
    return (x-mean)/std

def cache(root,local):
    records=json.loads((root/'development_image_manifest.json').read_text())
    array=np.memmap(local/'development_pixels.uint8',mode='r',dtype=np.uint8,shape=(len(records),3,512,512,3))
    with np.load(root/'teacher_and_targets.npz',allow_pickle=False) as z:targets={k:z[k].copy() for k in z.files}
    assert np.array_equal(targets['item_id'],[r['item_id'] for r in records])
    return records,array,targets

def prepare(root,local):
    p=context(root);local.mkdir(exist_ok=True);started=time.perf_counter()
    metadata=json.loads(Path(p['source_metadata']).read_text());folds=json.loads(Path(p['source_folds']).read_text())
    src=module('locked_source_entries',p['source_entry_helper'])
    sourceplan=json.loads(Path(p['source_plan']).read_text())
    sourcehash=json.loads(Path(p['prediction_hashes']).read_text())
    records={};teachers={};ground={}
    for folder,cohort,model,group,condition,path in src.dataset_entries(sourceplan):
        if model!='candidate300':continue
        assert sha(path)==sourcehash[str(path)]
        with np.load(path,allow_pickle=False) as z:
            for i,identifier in enumerate(z['sample_id']):
                key=f'{group}:{int(identifier)}';teachers.setdefault(key,{})[condition]=z['logits'][i].copy()
                if group.startswith('positive'):ground[key]=z['gt_bits'][i].tolist()
    for index,source in enumerate(p['image_sources']):
        archive=Path(p['previous_local'])/f'images_{index}.zip'
        if not archive.exists():
            archive=local/f'source_{index}.zip';shutil.copy2(source['archive'],archive)
        assert sha(archive)==source['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            for r in json.loads(Path(source['manifest']).read_text()):
                group=r.get('group',r.get('source'))
                group={'negative':'generated_negative','generated_ngc':'generated_negative','natural_photo':'natural_negative'}.get(group,group)
                key=f"{group}:{int(r['sample_id'])}";assert key in metadata and key not in records
                payload=z.read(r['relative']);assert hashlib.sha256(payload).hexdigest()==r['sha256']
                path=local/'development_pngs'/group/f"{int(r['sample_id'])}.png";path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
                record=dict(metadata[key],item_id=key,fold=folds[key],role=ROLES[folds[key]],path=str(path),sha256=r['sha256'])
                record['gt_bits']=ground[key] if key in ground else None;records[key]=record
        print('MULTITASK_SOURCE_IMAGES',len(records),flush=True)
    assert set(records)==set(metadata) and len(records)==4456
    rows=[records[key] for key in sorted(records)]
    for kind in ['content_key']:
        assigned={}
        for r in rows:assigned.setdefault(r[kind],set()).add(r['role'])
        assert all(len(a)==1 for a in assigned.values())
    assigned={}
    for r in rows:
        if r['user_id']>=0:assigned.setdefault((r['k'],r['user_id']),set()).add(r['role'])
    assert all(len(a)==1 for a in assigned.values())
    save(root/'development_image_manifest.json',rows)
    target=np.array([r['gt_bits'] if r['gt_bits'] is not None else [0]*32 for r in rows],dtype=np.uint8)
    teacher=np.stack([np.stack([teachers[r['item_id']][c] for c in CONDITIONS]) for r in rows]).astype(np.float32)
    np.savez_compressed(root/'teacher_and_targets.npz',item_id=np.array([r['item_id'] for r in rows]),gt_bits=target,teacher=teacher)
    mmap=np.memmap(local/'development_pixels.uint8',mode='w+',dtype=np.uint8,shape=(4456,3,512,512,3))
    helper=module('original_transform',p['transform_helper'])
    for i,r in enumerate(rows):
        for j,c in enumerate(CONDITIONS):
            mmap[i,j]=pixels(r['path'],c)
            if i<2:assert np.array_equal(mmap[i,j].astype(np.float32)/255,helper.transform(r['path'],c))
        if (i+1)%512==0:print('ACTUAL_JPEG_CACHE',i+1,4456,round(time.perf_counter()-started,1),flush=True)
    mmap.flush();del mmap
    # Lock the future evaluation fingerprints and photo IDs before fitting any neural parameters.
    ids=np.arange(2420,3444);assert len(ids)==p['test_per_group']
    for k in [11,16]:
        used={r['user_id'] for r in rows if r['k']==k}
        unused=np.array(sorted(set(range(2**k))-used),dtype=np.uint32)
        users=np.random.default_rng(2026102900+k).choice(unused,size=len(ids),replace=False)
        messages=((users[:,None]>>np.arange(k-1,-1,-1))&1).astype(np.uint8)
        generator=np.load(Path(p['study'])/f'bch31_{k}_p/generator_matrix.npz')['generator']
        words=messages@generator%2;words=np.column_stack([words,words.sum(axis=1)%2]).astype(np.uint8)
        np.savez_compressed(root/f'future_test_inputs_k{k}.npz',sample_id=ids,user_id=users,user_bits=messages,gt_bits=words)
        assert not set(users)&used
    oldphotos=set();oldphotohashes=set()
    for name in p['prior_photo_manifests']:
        for r in json.loads(Path(name).read_text())['records']:
            oldphotos.add(int(r['coco_id']));oldphotohashes.add(r['sha256'])
    with zipfile.ZipFile(p['photo_catalog']) as z:catalog=json.loads(z.read('annotations/image_info_test2014.json'))['images']
    available=sorted([r for r in catalog if int(r['id']) not in oldphotos],key=lambda r:r['id'])
    selected=np.random.default_rng(2026102901).choice(len(available),1024,replace=False)
    photos=[available[int(i)] for i in selected]
    save(root/'future_test_photo_selection.json',{'seed':2026102901,'records':photos,'prior_excluded':sorted(oldphotos),'prior_photo_hashes':sorted(oldphotohashes),'before_neural_training':True})
    counts={role:{g:sum(r['role']==role and r['group']==g for r in rows) for g in GROUPS} for role in ['fit','selection','calibration']}
    save(root/'preparation.json',{'complete':True,'source_images':4456,'role_counts':counts,'transformed_views':13368,
         'cached_pixel_bytes':(local/'development_pixels.uint8').stat().st_size,'cache_sha256':sha(local/'development_pixels.uint8'),
         'content_and_repeated_true_users_do_not_cross_roles':True,'future_test_users_disjoint_from_all_development_users':True,
         'future_test_not_generated_or_inferred':True,'six_JPEG_transform_controls_bitwise':True,'wall_seconds':time.perf_counter()-started})
    save(root/'data_lock.json',{name:sha(root/name) for name in ['development_image_manifest.json','teacher_and_targets.npz','future_test_inputs_k11.npz','future_test_inputs_k16.npz','future_test_photo_selection.json']})
    context(root);print('NEGATIVE_AWARE_INPUTS_LOCKED',counts,flush=True)

def assert_data(root):
    for name,value in json.loads((root/'data_lock.json').read_text()).items():assert sha(root/name)==value,name

class BCH:
    def __init__(self,p):
        self.books={};self.tau={};self.original=module('original_bch_decoder',p['syndrome_helper'])
        policies=json.loads(Path(p['old_policy']).read_text())
        for k in [11,16]:
            source=Path(p['study'])/f'bch31_{k}_p';policy=json.loads((source/'decoder_policy.json').read_text())
            with np.load(source/'bounded_distance_syndrome_table.npz') as z:table={name:z[name].copy() for name in z.files}
            generator=np.load(source/'generator_matrix.npz')['generator']
            self.books[k]=(source,policy,table,generator)
            self.tau[k]=policies['policies']['strict_zero_observed'][str(k)]['threshold']
    def decide(self,logits,k,audit=False):
        source,policy,table,generator=self.books[k]
        # Use the archived sigmoid >0.5 rule, including exact floating point ties.
        import torch
        received=(torch.from_numpy(logits).sigmoid().numpy()>.5).astype(np.uint8)
        words=np.sum(received[:,:31].astype(np.uint64)*(np.uint64(1)<<np.arange(30,-1,-1,dtype=np.uint64)),axis=1).astype(np.uint32)
        syndrome=words.copy();degree=31-k
        for exponent in range(30,degree-1,-1):
            select=(syndrome&np.uint32(1<<exponent))!=0;syndrome[select]^=np.uint32(policy['generator_polynomial_integer']<<(exponent-degree))
        positions=np.searchsorted(table['syndrome'],syndrome);safe=np.minimum(positions,len(table['syndrome'])-1)
        valid=(positions<len(table['syndrome']))&(table['syndrome'][safe]==syndrome)
        corrected=words^np.where(valid,table['error_word'][safe],0).astype(np.uint32)
        word31=((corrected[:,None]>>np.arange(30,-1,-1))&1).astype(np.uint8);message=word31[:,:k]
        assert np.array_equal(message[valid]@generator%2,word31[valid])
        word32=np.column_stack([word31,word31.sum(axis=1)%2]);distance=(word32!=received).sum(axis=1)
        ecc=valid&(distance<=policy['t']);score=-np.logaddexp(0,-(2*word32.astype(float)-1)*logits.astype(float)).mean(axis=1)
        user=np.sum(message.astype(np.uint64)*(np.uint64(1)<<np.arange(k-1,-1,-1,dtype=np.uint64)),axis=1).astype(np.int64)
        if audit:
            old=self.original.decide(received,source)
            assert np.array_equal(old[0],message) and np.array_equal(old[1],valid) and np.array_equal(old[2],distance) and np.array_equal(old[3],ecc)
        return {'candidate_user':user,'ecc':ecc,'bit_gate':ecc&(score>=self.tau[k]),'candidate_loglikelihood':score,'distance32':distance,'pred_bits':received}

def views_infer(model,records,mmap,targets,role,batch=8):
    import torch
    selected=[i for i,r in enumerate(records) if r['role']==role]
    output={'index':[],'condition':[],'logits':[],'presence':[]}
    model.eval()
    with torch.no_grad():
        for c in range(3):
            for start in range(0,len(selected),batch):
                ix=np.array(selected[start:start+batch]);b,s=model(normalized(torch,mmap[ix,c]))
                output['index'].extend(ix.tolist());output['condition'].extend([CONDITIONS[c]]*len(ix))
                output['logits'].append(b.cpu().numpy());output['presence'].append(s.cpu().numpy())
    return {'index':np.array(output['index']),'condition':np.array(output['condition']),
            'logits':np.concatenate(output['logits']),'presence':np.concatenate(output['presence'])}

def decisions(records,raw,bch,audit=False):
    data={key:value.copy() for key,value in raw.items()}
    groups=np.array([records[i]['group'] for i in raw['index']]);data['group']=groups
    data['sample_id']=np.array([records[i]['sample_id'] for i in raw['index']])
    data['item_id']=np.array([records[i]['item_id'] for i in raw['index']])
    data['gt_user_id']=np.array([records[i]['user_id'] for i in raw['index']])
    data['gt_bits']=np.array([records[i]['gt_bits'] if records[i]['gt_bits'] is not None else [0]*32 for i in raw['index']],dtype=np.uint8)
    result={}
    for k in [11,16]:
        mask=(groups==f'positive_k{k}')|~np.char.startswith(groups,'positive')
        d={key:v[mask] for key,v in data.items()};d.update(bch.decide(d['logits'],k,audit))
        positive=d['group']==f'positive_k{k}';d['positive']=positive
        d['correct_candidate']=positive&(d['candidate_user']==d['gt_user_id'])&d['ecc']
        d['bad_candidate']=~d['correct_candidate']
        result[k]=d
    return result

def metrics(d,accept):
    rows=[]
    for condition in CONDITIONS:
        for group in [f"positive_k{11 if np.any(d['group']=='positive_k11') else 16}",'generated_negative','natural_negative']:
            m=(d['condition']==condition)&(d['group']==group)
            rows.append({'condition':condition,'group':group,'images':int(m.sum()),'correct':int((m&accept&d['correct_candidate']).sum()),
                         'bad':int((m&accept&d['bad_candidate']).sum()),'reject':int((m&~accept).sum())})
    return {'correct':sum(r['correct'] for r in rows),'bad':sum(r['bad'] for r in rows),
            'correct_by_condition':{c:sum(r['correct'] for r in rows if r['condition']==c) for c in CONDITIONS},'cells':rows}

def fit_presence_rule(d):
    bad=d['bit_gate']&d['bad_candidate']
    threshold=max(0.,float(np.nextafter(d['presence'][bad].max(),np.inf))) if bad.any() else 0.
    accept=d['bit_gate']&(d['presence']>=threshold)
    assert not (accept&d['bad_candidate']).any()
    return threshold,metrics(d,accept),accept

def save_decisions(root,name,ds,thresholds=None):
    import torch
    for k,d in ds.items():
        v=dict(d)
        v['probabilities']=torch.from_numpy(d['logits']).sigmoid().numpy()
        v['gt_bits_valid']=d['positive'].copy()
        v['bit_errors']=np.where(d['positive'],(d['pred_bits']!=d['gt_bits']).sum(axis=1),-1)
        if thresholds is not None:v['final_accept']=d['bit_gate']&(d['presence']>=thresholds[k])
        np.savez_compressed(root/f'{name}_k{k}.npz',**v)

def train(root,local):
    p=context(root);assert_data(root);torch=configure(True)
    from torch import nn
    assert sha(local/'development_pixels.uint8')==json.loads((root/'preparation.json').read_text())['cache_sha256']
    records,mmap,targets=cache(root,local);bch=BCH(p);model=make_model(p)
    initial={name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
    frozen=[name for name,v in model.named_parameters() if not v.requires_grad]
    buffers=[name for name in initial if any(t in name for t in ['running_mean','running_var','num_batches_tracked'])]
    # Reproduce archived bit outputs before making any optimizer update.
    controls=[]
    with torch.no_grad():
        for c in range(3):
            ix=np.array([i for i,r in enumerate(records) if r['group']=='positive_k11'][:4])
            logits,_=model(normalized(torch,mmap[ix,c]));values=logits.cpu().numpy();reference=targets['teacher'][ix,c]
            assert np.allclose(values,reference,atol=1e-4,rtol=1e-4)
            assert np.array_equal(torch.from_numpy(values).sigmoid().numpy()>.5,torch.from_numpy(reference).sigmoid().numpy()>.5)
            controls.append({'condition':CONDITIONS[c],'images':4,'bits_match':True,'max_logit_difference':float(np.abs(values-reference).max())})
    save(root/'initial_model_reproduction.json',controls);print('INITIAL_MULTITASK_BIT_OUTPUTS_REPRODUCED12',flush=True)
    selection_ix=np.array([i for i,r in enumerate(records) if r['role']=='selection'])
    ref={'index':np.tile(selection_ix,3),'condition':np.repeat(CONDITIONS,len(selection_ix)),
         'logits':np.concatenate([targets['teacher'][selection_ix,c] for c in range(3)]),'presence':np.zeros(3*len(selection_ix),np.float32)}
    reference=decisions(records,ref,bch,audit=True)
    baseline={k:metrics(d,d['bit_gate']) for k,d in reference.items()};save(root/'selection_reference_candidate300.json',baseline)
    pools={g:np.array([i for i,r in enumerate(records) if r['role']=='fit' and r['group']==g]) for g in GROUPS}
    paired={r['sample_id']:i for i,r in enumerate(records) if r['role']=='fit' and r['group']=='generated_negative'}
    assert all(records[i]['sample_id'] in paired for g in GROUPS[:2] for i in pools[g])
    groups=[{'params':[v for n,v in model.named_parameters() if n.startswith(prefix) and v.requires_grad],'lr':rate,'initial_rate':rate}
            for prefix,rate in [('backbone.layer3.',2e-6),('backbone.layer4.',5e-6),('bit_head.',1e-5),('presence_head.',1e-4)]]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    random=np.random.default_rng(20261029);history=[];assessments=[];best=None;best_rank=None
    started=time.perf_counter()
    bce=nn.BCEWithLogitsLoss();trainable=[v for v in model.parameters() if v.requires_grad]
    for step in range(1,p['steps']+1):
        assert time.perf_counter()-started<p['training_timeout_seconds']-120,'Operational training timeout'
        positive=np.concatenate([random.choice(pools[g],4,replace=False) for g in GROUPS[:2]])
        matched=np.array([paired[records[i]['sample_id']] for i in positive])
        ix=np.concatenate([positive,matched,random.choice(pools['generated_negative'],4,replace=False),random.choice(pools['natural_negative'],4,replace=False)])
        assert len(ix)==24 and all(records[i]['role']=='fit' for i in ix)
        c=[0,1,2,2][(step-1)%4]
        model.eval();model.presence_head.train();optimizer.zero_grad(set_to_none=True)
        bits,presence=model(normalized(torch,mmap[ix,c]))
        target=torch.from_numpy(targets['gt_bits'][positive].copy()).float().cuda()
        teacher=torch.from_numpy(targets['teacher'][positive,c].copy()).cuda()
        loss_bits=bce(bits[:8],target)
        loss_presence=.5*bce(presence[:8],torch.ones_like(presence[:8]))+.25*bce(presence[8:20],torch.zeros_like(presence[8:20]))+.25*bce(presence[20:],torch.zeros_like(presence[20:]))
        neutral=.5*bce(bits[8:20],torch.full_like(bits[8:20],.5))+.5*bce(bits[20:],torch.full_like(bits[20:],.5))
        retention=(bits[:8]-teacher).square().mean()
        paired_loss=torch.nn.functional.softplus(1-presence[:8]+presence[8:16]).mean()
        loss=2*loss_bits+loss_presence+.05*neutral+.05*retention+.2*paired_loss
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(trainable,1.)
        warm=min(1.,step/100);progress=max(0.,(step-100)/(p['steps']-100));factor=warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for g in optimizer.param_groups:g['lr']=g['initial_rate']*factor
        optimizer.step()
        history.append({'step':step,'condition':CONDITIONS[c],'fit_indices':ix.tolist(),'loss':float(loss.item()),'bits':float(loss_bits.item()),
                        'presence':float(loss_presence.item()),'negative_uncertainty':float(neutral.item()),'teacher_MSE':float(retention.item()),'paired_presence':float(paired_loss.item())})
        if step%100==0:
            save(root/'training_history.json',history)
            print('NEGATIVE_AWARE_TRAINING',step,p['steps'],'loss',round(float(loss.item()),5),'elapsed',round(time.perf_counter()-started,1),flush=True)
        if step%400==0:
            raw=views_infer(model,records,mmap,targets,'selection');ds=decisions(records,raw,bch)
            thresholds={};measures={};retain={}
            for k,d in ds.items():
                thresholds[k],measures[k],_=fit_presence_rule(d)
                retain[k]={c:measures[k]['correct_by_condition'][c]/max(1,baseline[k]['correct_by_condition'][c]) for c in CONDITIONS}
            primary=min(retain[11].values());secondary=min(retain[16].values())
            rank=(primary>=.95,primary,measures[11]['correct'],secondary,measures[16]['correct'])
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            weights=root/f'checkpoint_step{step:04d}.pth';torch.save(state,weights)
            row={'step':step,'selection_only_presence_thresholds':thresholds,'selection_metrics':measures,'retention_vs_initial_candidate300':retain,
                 'rank':list(rank),'no_calibration_or_future_test_inference':True}
            assessments.append(row);save(root/'selection_history.json',assessments)
            if best_rank is None or rank>best_rank:
                best_rank=rank;best=step;torch.save(state,root/'selected_multitask.pth')
                save_decisions(root,'selection_selected',ds,thresholds)
            torch.save({'state_dict':state,'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':random.bit_generator.state,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best_step':best},root/'last_resumable_checkpoint.pth')
            print('NEGATIVE_AWARE_SELECTION',json.dumps({'step':step,'primary_correct':measures[11]['correct'],'primary_retention':retain[11],'secondary_retention':retain[16],'best_step':best}),flush=True)
    for n in frozen+buffers:assert torch.equal(model.state_dict()[n].cpu(),initial[n]),n
    assert_data(root);context(root)
    save(root/'selection_locked.json',{'complete':True,'selected_step':best,'selected_weights_sha256':sha(root/'selected_multitask.pth'),
         'steps_completed':p['steps'],'ranking':list(best_rank),'calibration_and_test_not_used':True,'frozen_parameters_unchanged':len(frozen),
         'batchnorm_buffers_unchanged':True,'initial_generator_and_mapping_unchanged':True,'wall_seconds':time.perf_counter()-started})
    print('NEGATIVE_AWARE_TRAINING_COMPLETE',best,flush=True)

def calibrate(root,local):
    p=context(root);assert_data(root);torch=configure()
    lock=json.loads((root/'selection_locked.json').read_text());assert lock['complete'] and sha(root/'selected_multitask.pth')==lock['selected_weights_sha256']
    assert not (root/'future_test_generation.json').exists()
    records,mmap,targets=cache(root,local);model=make_model(p);model.load_state_dict(torch.load(root/'selected_multitask.pth',map_location='cpu'),strict=True)
    bch=BCH(p);raw=views_infer(model,records,mmap,targets,'calibration');ds=decisions(records,raw,bch,True)
    ix=np.array([i for i,r in enumerate(records) if r['role']=='calibration'])
    ref={'index':np.tile(ix,3),'condition':np.repeat(CONDITIONS,len(ix)),'logits':np.concatenate([targets['teacher'][ix,c] for c in range(3)]),'presence':np.zeros(len(ix)*3,np.float32)}
    reference=decisions(records,ref,bch,True)
    thresholds={};calibration={}
    for k,d in ds.items():
        threshold,measured,accept=fit_presence_rule(d);thresholds[k]=threshold
        baseline=metrics(reference[k],reference[k]['bit_gate'])
        retention={c:measured['correct_by_condition'][c]/max(1,baseline['correct_by_condition'][c]) for c in CONDITIONS}
        calibration[k]={'threshold':threshold,'new_joint':measured,'reference_candidate300':baseline,'retention':retention,
                        'eligible_for_later_adoption_review':measured['bad']==0 and min(retention.values())>=.95}
    save_decisions(root,'calibration',ds,thresholds)
    save(root/'locked_presence_policy.json',{'complete':True,'selected_weights_sha256':lock['selected_weights_sha256'],'per_code':calibration,
          'rule':'existing BCH radius and fixed original per-code candidate bit-likelihood threshold AND trained presence logit >= independently calibrated threshold',
          'no_future_test_generated_or_observed':True,'no_active_policy_replacement':True})
    save(root/'calibration_verification.json',{'complete':True,'rules_locked_before_test_generation':True,'original_syndrome_decisions_match':True})
    print('NEGATIVE_AWARE_CALIBRATION_LOCKED',json.dumps(calibration),flush=True)

def newtest(root,local):
    p=context(root);assert_data(root);configure();started=time.perf_counter()
    import torch
    from PIL import Image
    from torchvision import transforms
    from diffusers import AutoencoderKL
    sys.path.insert(0,p['repo']);from attribution import MappingNetwork
    from customization import customize_vae_decoder
    assert json.loads((root/'calibration_verification.json').read_text())['complete']
    policy_hash=sha(root/'locked_presence_policy.json');assert not (root/'future_test_generation.json').exists()
    latentdir=local/'future_latents';latentdir.mkdir(exist_ok=False);wanted=set(range(2420,3444))|{1280,1281,1282,1283};latentrows=[]
    with tarfile.open(p['latent_archive'],'r|gz') as z:
        for member in z:
            name=Path(member.name)
            if not member.isfile() or name.suffix!='.pth' or name.parent.name!='COCO' or not name.stem.isdecimal():continue
            identifier=int(name.stem)
            if identifier not in wanted:continue
            path=latentdir/f'{identifier}.pth'
            with z.extractfile(member) as incoming,path.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
            latentrows.append({'sample_id':identifier,'member':member.name,'sha256':sha(path)});wanted.remove(identifier)
            if not wanted:break
    assert not wanted;save(root/'future_latent_manifest.json',latentrows)
    photos=json.loads((root/'future_test_photo_selection.json').read_text());oldhash=set(photos['prior_photo_hashes'])
    image_rows=[];originaldir=local/'future_photo_originals';originaldir.mkdir()
    resize=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512)])
    def download(row):
        name=f"COCO_test2014_{int(row['id']):012d}.jpg";assert name==row['file_name']
        url='http://images.cocodataset.org/test2014/'+name;last=None
        for attempt in range(3):
            try:
                payload=urllib.request.urlopen(url,timeout=30).read();digest=hashlib.sha256(payload).hexdigest();assert digest not in oldhash
                target=local/'future_pngs/natural_negative'/f"{int(row['id'])}.png";target.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(io.BytesIO(payload)) as f:assert f.format=='JPEG';image=resize(f.convert('RGB'));image.save(target)
                (originaldir/name).write_bytes(payload)
                return {'group':'natural_negative','sample_id':int(row['id']),'user_id':-1,'gt_bits':None,'path':str(target),'sha256':sha(target),
                        'original_sha256':digest,'url':url}
            except Exception as error:last=repr(error)
        raise RuntimeError(str(row['id'])+' '+str(last))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for row in pool.map(download,photos['records']):
            image_rows.append(row)
            if len(image_rows)%128==0:print('FUTURE_NATURAL_PHOTOS',len(image_rows),1024,flush=True)
    assert len({r['original_sha256'] for r in image_rows})==1024
    control_manifest=json.loads((Path(p['fixed_root'])/'image_manifest.json').read_text())
    with zipfile.ZipFile(Path(p['previous_local'])/'images_1.zip') as controls:
        for group in GROUPS[:3]:
            reference_group='negative' if group=='generated_negative' else group
            if group=='generated_negative':
                vae=AutoencoderKL.from_config(str(Path(p['pristine_vae'])/'config.json'))
                state=torch.load(Path(p['pristine_vae'])/'diffusion_pytorch_model.bin',map_location='cpu')
                assert not any('affine' in n for n in state);vae.load_state_dict(state,strict=True);del state;mapping=None;inputs=None
            else:
                vae=customize_vae_decoder(AutoencoderKL.from_config(str(Path(p['snapshot'])/'final_pipeline/vae/config.json')),128,1)
                vae.load_state_dict(torch.load(Path(p['snapshot'])/'checkpoint-3000/pytorch_model.bin',map_location='cpu'),strict=True)
                mapping=MappingNetwork(32,128,num_layers=2);mapping.load_state_dict(torch.load(Path(p['snapshot'])/'weights/mapping_network.pth',map_location='cpu'),strict=True)
                mapping.requires_grad_(False).eval().float().cuda()
                inputs=np.load(root/f"future_test_inputs_k{int(group.removeprefix('positive_k'))}.npz")
            vae.requires_grad_(False).eval().float().cuda()
            def decode(ids,bits):
                latent=torch.stack([torch.load(latentdir/f'{int(i)}.pth',map_location='cpu') for i in ids]).float().cuda()
                value=vae.decode(latent/.18215).sample if bits is None else vae.decode(latent/.18215,mapping(torch.from_numpy(bits.copy()).float().cuda()+1e-8)).sample
                value=(value.clamp(-1,1)/2+.5).clamp(0,1)
                return (value*255+.5).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
            refbits=None
            if mapping is not None:
                refbits=np.load(Path(p['fixed_root'])/'inputs'/f'{reference_group}.npz')['gt_bits'][:4].copy()
            values=decode(range(1280,1284),refbits)
            for j,i in enumerate(range(1280,1284)):
                r=next(r for r in control_manifest if r['group']==reference_group and r['sample_id']==i)
                payload=controls.read(r['relative']);assert hashlib.sha256(payload).hexdigest()==r['sha256']
                with Image.open(io.BytesIO(payload)) as f:assert np.array_equal(values[j],np.asarray(f.convert('RGB')))
            print('GENERATOR_PIXEL_REFERENCE_EXACT',group,4,flush=True)
            directory=local/'future_pngs'/group;directory.mkdir(parents=True,exist_ok=False)
            for start in range(0,1024,4):
                ids=np.arange(2420,3444)[start:start+4];bits=None if inputs is None else inputs['gt_bits'][start:start+4]
                values=decode(ids,bits)
                for j,i in enumerate(ids):
                    path=directory/f'{int(i)}.png';Image.fromarray(values[j]).save(path)
                    image_rows.append({'group':group,'sample_id':int(i),'user_id':-1 if inputs is None else int(inputs['user_id'][start+j]),
                                       'gt_bits':None if inputs is None else bits[j].tolist(),'path':str(path),'sha256':sha(path)})
                if (start+4)%128==0:print('FUTURE_GENERATED_IMAGES',group,start+4,1024,'elapsed',round(time.perf_counter()-started,1),flush=True)
            del vae,mapping;torch.cuda.empty_cache()
    assert len(image_rows)==4096;save(root/'future_test_image_manifest.json',image_rows)
    archive=local/'future_test_source_images.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for r in image_rows:z.write(r['path'],arcname=f"{r['group']}/{r['sample_id']}.png")
    shutil.copy2(archive,root/archive.name);assert sha(archive)==sha(root/archive.name)
    assert sha(root/'locked_presence_policy.json')==policy_hash
    save(root/'future_test_generation.json',{'complete':True,'source_images':4096,'per_group':1024,'archived_sha256':sha(root/archive.name),
         'generator_control_pixels_exact':12,'new_users_disjoint_from_all_development':True,'fresh_natural_photos':1024,
         'latent_content_previously_in_original_baseline5000':True,'new_branch_did_not_train_or_select_on_test_content':True,'policy_hash_before_generation':policy_hash,
         'wall_seconds':time.perf_counter()-started})
    context(root);print('FUTURE_TEST_SOURCES_SAVED',4096,flush=True)

def test(root,local):
    p=context(root);assert_data(root);torch=configure();assert json.loads((root/'future_test_generation.json').read_text())['complete']
    policy=json.loads((root/'locked_presence_policy.json').read_text());phash=sha(root/'locked_presence_policy.json')
    assert sha(root/'selected_multitask.pth')==policy['selected_weights_sha256']
    rows=json.loads((root/'future_test_image_manifest.json').read_text())
    records=[dict(r,item_id=f"{r['group']}:{r['sample_id']}",role='test') for r in rows]
    features=np.memmap(local/'future_pixels.uint8',mode='w+',dtype=np.uint8,shape=(4096,3,512,512,3))
    for i,r in enumerate(records):
        assert sha(r['path'])==r['sha256']
        for c,condition in enumerate(CONDITIONS):features[i,c]=pixels(r['path'],condition)
    features.flush();bch=BCH(p)
    from torchvision.models import resnet50
    from torch import nn
    started=time.perf_counter()
    for name in ['old200','candidate300','negative_aware']:
        if name=='negative_aware':
            model=make_model(p);model.load_state_dict(torch.load(root/'selected_multitask.pth',map_location='cpu'),strict=True)
        else:
            backbone=resnet50(weights=None);backbone.fc=nn.Linear(2048,32)
            backbone.load_state_dict(torch.load(p['reference_extractors'][name],map_location='cpu'),strict=True);backbone.requires_grad_(False).eval().float().cuda()
            class Reference(nn.Module):
                def __init__(self,base):super().__init__();self.base=base
                def forward(self,x):
                    b=self.base(x);return b,torch.zeros(len(b),device=b.device)
            model=Reference(backbone)
        raw=views_infer(model,records,features,None,'test');ds=decisions(records,raw,bch,True)
        thresholds={k:policy['per_code'][str(k)]['threshold'] for k in [11,16]} if name=='negative_aware' else {11:-float('inf'),16:-float('inf')}
        save_decisions(root,'future_test_'+name,ds,thresholds)
        del model
        if name!='negative_aware':del backbone
        torch.cuda.empty_cache();print('FUTURE_TEST_MODEL_COMPLETE',name,'elapsed',round(time.perf_counter()-started,1),flush=True)
    assert sha(root/'locked_presence_policy.json')==phash;context(root)
    save(root/'test_inference_verification.json',{'complete':True,'source_images':4096,'views':12288,'model_image_inferences':36864,
         'models':3,'original_syndrome_decisions_agree':True,'policy_locked_before_any_test_outputs':True,'policy_hash_unchanged':phash,
         'no_test_based_retuning':True,'wall_seconds':time.perf_counter()-started})

def report(root,local):
    p=context(root);assert json.loads((root/'test_inference_verification.json').read_text())['complete']
    allrows=[];summaries=[]
    for k in [11,16]:
        outputs={}
        for model in ['old200','candidate300','negative_aware']:
            with np.load(root/f'future_test_{model}_k{k}.npz') as z:d={n:z[n].copy() for n in z.files}
            measured=metrics(d,d['final_accept']);outputs[model]=measured
            for row in measured['cells']:
                assert row['images']==1024 and row['correct']+row['bad']+row['reject']==1024
                allrows.append({'model':model,'k':k,**row})
        new,reference=outputs['negative_aware'],outputs['candidate300']
        retention={c:new['correct_by_condition'][c]/max(1,reference['correct_by_condition'][c]) for c in CONDITIONS}
        cal=json.loads((root/'locked_presence_policy.json').read_text())['per_code'][str(k)]
        promising=bool(cal['eligible_for_later_adoption_review'] and new['bad']<reference['bad'] and new['bad']==0 and min(retention.values())>=.95)
        summaries.append({'k':k,'primary':k==11,'models':outputs,'correct_retention_vs_candidate300':retention,
                          'meets_predeclared_candidate_criteria':promising,'not_automatically_adopted':True})
    save(root/'summary.json',summaries)
    with (root/'test_cell_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    primary=summaries[0];table='<table><tr><th>提取器</th><th>正确恢复记录</th><th>坏接受记录</th><th>PNG正确/1024</th><th>JPEG444正确/1024</th><th>JPEG420正确/1024</th></tr>'
    for model,values in primary['models'].items():
        row=[model,values['correct'],values['bad']]+[values['correct_by_condition'][c] for c in CONDITIONS]
        table+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>'
    table+='</table>'
    message='新候选达到本轮预先固定的候选要求；仍需进一步独立复核后再替换工作配置。' if primary['meets_predeclared_candidate_criteria'] else '新候选未达到本轮预先固定的采用条件，保留原工作配置，保存训练结果用于分析。'
    details=[message,
        '本轮实际完成2400步专门训练：从+300提取器初始化，训练layer3/layer4、32位输出和新水印存在性头；生成器、映射网络、早期层及BN统计保持不变。正样本指纹恢复＋存在性分类＋无水印低置信度＋教师保留＋同内容配对损失，实际JPEG Q90 444/420参与训练。',
        '旧4456图按已有内容/重复真用户连通四折划分：0/1仅拟合，2仅选择六个固定检查点之一，3只在模型固定后定门槛。旧批次都曾被查看，仍属于开发过程；没有把旧测试集重新称为未见数据。',
        '前瞻对照4096源图：BCH11、BCH16、无水印生成图、自然照片各1024，三种压缩版本共12288视图；三个模型均用同图对照。latent2420..3443未用于当前适配训练与选择，但曾进入最早baseline5000评估，不能称为从未使用过的外部测试集。新分配的真用户与本轮4456图开发集合的真用户不重合；自然照片排除此前已用ID和文件hash。',
        '坏接受包括带水印图认错用户，以及无水印图接受任意用户；总数含同图压缩变体，不能作为独立总体FPR。按来源/条件分母见test_cell_results.csv。即使某条件1024张观察到零错，也不证明总体零错；零错的单侧95%二项上界约0.292%，且该上界依赖独立同分布假设，不合并三个压缩版本。',
        '新规则使用原BCH半径和原位似然门槛，再加独立校准的存在性门槛。校准阈值固定为至少0并高于全部校准坏接受分数；PNG、444、420各条件正确恢复须保留初始+300的95%，前瞻对照须零坏接受且少于+300才标记有潜力。没有依据前瞻结果调门槛、换检查点或训练。',
        '主比较BCH(31,11)+P；BCH(31,16)+P保留完整对照。原始3000-step生成器未更新，图像质量生成链保持原状。冻结Python/PyTorch/Pillow等环境未安装或升级包。权重、训练记录、恢复检查点、输入图像zip、逐样本logits/概率可由logits重算、预测bit、归属决策及hash保留Drive。'
    ]
    css='<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1300px;margin:20px auto;padding:18px;background:#f3f8fc;color:#17394e}table{border-collapse:collapse;width:100%}td,th{border:1px solid #b4c7d5;padding:8px}th{background:#dfebf5}p{line-height:1.6}</style>'
    title='负样本感知专门训练：锁定模型后的前瞻对照'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title>'+css+'<h1>'+title+'</h1>'+table+''.join('<p>'+html.escape(x)+'</p>' for x in details)+'</html>'
    (root/'RESULTS_zh.html').write_text(page)
    brief='<div class="negativebrief"><style>.negativebrief{background:#f3f8fc;padding:18px}.negativebrief *{color:#17394e!important}.negativebrief table{border-collapse:collapse;width:100%}.negativebrief td,.negativebrief th{border:1px solid #b4c7d5;padding:8px}</style><h2>'+title+'</h2>'+table+'<p><b>'+message+'</b></p><p>实际训练2400步；新渲染4096源图，三压缩条件、两编码对照。latents曾进入早期baseline评估，本轮新用户和适配数据隔离。记录含同图压缩变体。</p><p>现用配置未替换；训练检查点、原始输出、源图和恢复入口已存Drive。</p></div>'
    (root/'BRIEF_zh.html').write_text(brief)
    (root/'CONTINUE_HERE_zh.md').write_text('# '+title+'\n\n'+'\n\n'.join(details)+'\n\n原冻结环境恢复入口：<ORIGINAL_PRIVATE_DRIVE>/WOUAF_ECC/overnight_checkpoint_20261028T120231Z/restore_for_next_session.py\n先读本目录plan、summary、RESULTS及pipeline_status；不要Run all或重复开始训练。\n')
    context(root);assert_data(root)
    save(root/'delivery_verification.json',{'complete':True,'all54test_cells_verified':len(allrows)==54,'all_denominators_1024':True,
         'active_original_assets_unchanged':True,'no_test_retuning':True,'no_environment_changes':True,'generator_frozen':True})
    save(root/'decision.json',{'primary_meets_predeclared_candidate_criteria':primary['meets_predeclared_candidate_criteria'],'no_active_replacement':True,
         'next_step':'Audit and independently replicate the locked candidate before promotion.' if primary['meets_predeclared_candidate_criteria'] else 'Inspect per-source errors and retention; do not tune on this evaluation or automatically promote.'})
    print('NEGATIVE_AWARE_REPORT_COMPLETE',json.dumps(primary),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','train','calibrate','newtest','test','report']);a.add_argument('--root',type=Path,required=True);a.add_argument('--local',type=Path,required=True)
    args=a.parse_args();globals()[args.mode](args.root,args.local)

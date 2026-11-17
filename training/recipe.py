"""Portable implementation of the recorded paired, negative-aware recipe."""
import json
from pathlib import Path
import numpy as np
import torch
from ecc_wouaf.core import CONDITIONS,Extractor,Decoder,image_pixels,normalize_pixels,load_config,sha256

GROUPS=['positive_k11','positive_k16','generated_negative','natural_negative']

def save(path,value):
    p=Path(path);temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(p)

def dataset(directory):
    directory=Path(directory).resolve();records=json.loads((directory/'manifest.json').read_text())
    assert len({r['item_id'] for r in records})==len(records)
    with np.load(directory/'teacher_and_targets.npz',allow_pickle=False) as z:targets={n:z[n].copy() for n in z.files}
    assert np.array_equal(targets['item_id'],[r['item_id'] for r in records])
    content={};users={}
    for r in records:
        path=(directory/r['image']).resolve();assert path.is_relative_to(directory) and sha256(path)==r['sha256'],r['image']
        content.setdefault(r['content_key'],set()).add(r['role'])
        if r['user_id']>=0:users.setdefault((r['k'],r['user_id']),set()).add(r['role'])
    assert all(len(v)==1 for v in list(content.values())+list(users.values()))
    return records,targets

def initialize(path):
    model=Extractor();state=torch.load(path,map_location='cpu')
    assert 'fc.weight' in state and state['fc.weight'].shape==(32,2048),'Use the original WOUAF +300 initialization, not random or published multitask weights.'
    model.backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
    model.bit_head.load_state_dict({k.removeprefix('fc.'):v for k,v in state.items() if k.startswith('fc.')},strict=True)
    for name,value in model.named_parameters():value.requires_grad_(name.startswith(('backbone.layer3.','backbone.layer4.','bit_head.','presence_head.')))
    return model.eval().float().cuda(),state

def image_cache(directory,records,path):
    shape=(len(records),3,512,512,3)
    expected=int(np.prod(shape))
    if path.exists():
        assert path.stat().st_size==expected and path.with_suffix('.complete.json').exists(),'Incomplete derived cache; remove only this run cache before rebuilding.'
        return np.memmap(path,mode='r',dtype=np.uint8,shape=shape)
    cache=np.memmap(path,mode='w+',dtype=np.uint8,shape=shape)
    for i,r in enumerate(records):
        for c,condition in enumerate(CONDITIONS):cache[i,c]=image_pixels(Path(directory)/r['image'],condition)
        if (i+1)%512==0:print('PIXEL_CACHE',i+1,len(records),flush=True)
    cache.flush();save(path.with_suffix('.complete.json'),{'complete':True,'bytes':expected});return cache

def infer(model,records,cache,role,batch=8):
    selected=np.array([i for i,r in enumerate(records) if r['role']==role]);columns=[];presences=[]
    model.eval()
    with torch.no_grad():
        for c in range(3):
            for start in range(0,len(selected),batch):
                bits,presence=model(normalize_pixels(cache[selected[start:start+batch],c],'cuda'))
                columns.append(bits.cpu().numpy());presences.append(presence.cpu().numpy())
    return {'index':np.tile(selected,3),'condition':np.repeat(CONDITIONS,len(selected)),
            'logits':np.concatenate(columns),'presence':np.concatenate(presences)}

def decision_data(records,raw,decoder):
    groups=np.array([records[i]['group'] for i in raw['index']]);users=np.array([records[i]['user_id'] for i in raw['index']])
    results={}
    for k in [11,16]:
        selected=(groups==f'positive_k{k}')|~np.char.startswith(groups,'positive')
        d={key:v[selected] for key,v in raw.items()};d['group']=groups[selected]
        d['positive']=d['group']==f'positive_k{k}';d['user_id']=users[selected]
        decoded=decoder.decode(d['logits'],d['presence'],k)
        d.update(decoded);d['correct']=d['positive']&d['ecc_accept']&(d['candidate_user_id']==d['user_id'])
        d['bad']=~d['correct'];results[k]=d
    return results

def metrics(d,accept):
    return {'correct':int((accept&d['correct']).sum()),'bad':int((accept&d['bad']).sum()),
            'correct_by_condition':{c:int((accept&d['correct']&(d['condition']==c)).sum()) for c in CONDITIONS}}

def rule(d):
    bad=d['bit_gate']&d['bad'];threshold=max(0.,float(np.nextafter(d['presence'][bad].max(),np.inf))) if bad.any() else 0.
    accepted=d['bit_gate']&(d['presence']>=threshold)
    assert not (accepted&d['bad']).any()
    return threshold,metrics(d,accepted)

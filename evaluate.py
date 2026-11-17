"""Evaluate actual encoded files without recompression, fitting, or route overrides."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from ecc_wouaf import Predictor
from ecc_wouaf.core import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Directory containing manifest.json and native encoded files')
    p.add_argument('--version',choices=['v1','v2','v3'],default='v3')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--allow-unvalidated-runtime',action='store_true');a=p.parse_args()
    assert not a.out.exists(),'Use a new output directory.'
    rows=json.loads((a.data/'manifest.json').read_text());model=Predictor(a.version,a.device,not a.allow_unvalidated_runtime)
    assert len({r['image_id'] for r in rows})==len(rows)
    a.out.mkdir(parents=True);decisions={11:[],16:[]}
    with (a.out/'predictions.jsonl').open('w') as output:
        for i,r in enumerate(rows):
            path=(a.data/r['image']).resolve();assert path.is_relative_to(a.data.resolve())
            assert sha256(path)==r['sha256'],r['image']
            k=int(r['group'].removeprefix('positive_k')) if r['group'].startswith('positive_k') else 11
            value=model.predict(path,k)
            assert value['container']==r['container'] and value['jpeg_sampling']==r['jpeg_sampling']
            ks=[k] if r['group'].startswith('positive_k') else [11,16]
            branch=model.model(value['selected_branch'])
            for code in ks:
                d=branch.decoder.decode(np.asarray([value['logits']],np.float32),np.asarray([value['presence_logit']],np.float32),code)
                accepted=bool(d['accepted'][0]);positive=r['group']==f'positive_k{code}'
                correct=positive and accepted and int(d['candidate_user_id'][0])==r['user_id']
                record={**r,'version':a.version,'k':code,'accepted':accepted,'correct':correct,'bad':accepted and not correct,
                    'candidate_user_id':int(d['candidate_user_id'][0]),'user_id_if_accepted':int(d['candidate_user_id'][0]) if accepted else None,
                    'logits':value['logits'],'presence_logit':value['presence_logit'],'pred_bits':value['pred_bits'],
                    'probabilities':value['probabilities'],'selected_branch':value['selected_branch']}
                decisions[code].append(record);output.write(json.dumps(record)+'\n')
            if (i+1)%256==0:print('NATIVE_VIEWS',i+1,len(rows),flush=True)
    metrics=[]
    for k,items in decisions.items():
        for condition in sorted({r['condition'] for r in items}):
            for group in sorted({r['group'] for r in items}):
                selected=[r for r in items if r['condition']==condition and r['group']==group]
                if not selected:continue
                metrics.append({'version':a.version,'k':k,'condition':condition,'group':group,'images':len(selected),
                    'correct':sum(r['correct'] for r in selected),'bad':sum(r['bad'] for r in selected),'reject':sum(not r['accepted'] for r in selected)})
        np.savez_compressed(a.out/f'predictions_{a.version}_k{k}.npz',
            image_id=np.array([r['image_id'] for r in items]),group=np.array([r['group'] for r in items]),
            sample_id=np.array([r['sample_id'] for r in items]),scene_id=np.array([r.get('scene_id',-1) for r in items]),
            condition=np.array([r['condition'] for r in items]),gt_user_id=np.array([r['user_id'] for r in items]),
            gt_bits=np.array([r.get('gt_bits') or [0]*32 for r in items],np.uint8),
            gt_bits_valid=np.array([r.get('gt_bits') is not None for r in items]),
            logits=np.array([r['logits'] for r in items],np.float32),presence=np.array([r['presence_logit'] for r in items],np.float32),
            probabilities=np.array([r['probabilities'] for r in items],np.float32),pred_bits=np.array([r['pred_bits'] for r in items],np.uint8),
            candidate_user=np.array([r['candidate_user_id'] for r in items]),final_accept=np.array([r['accepted'] for r in items]))
    with (a.out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('EVALUATION_COMPLETE',len(rows),'native views; zero optimizer steps')
if __name__=='__main__':main()

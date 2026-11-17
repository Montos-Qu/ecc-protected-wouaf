"""Recompute real-error statistics and paired V1/V3 results without image inference."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import binomtest
from ecc_wouaf.core import Decoder,load_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1',type=Path,default=ROOT/'reports/independent')
    p.add_argument('--v3',type=Path,default=ROOT/'reports/independent')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    bootstrap=np.random.default_rng(410930000).integers(0,64,size=(10000,64))
    decoder=Decoder(load_config());cells=[];paired=[];errors=[];count=0;identical=True
    def load(folder,version,k):
        path=folder/f'predictions_{version}_k{k}.npz'
        if not path.exists():path=folder/f'predictions_{version.lower()}_k{k}.npz'
        with np.load(path,allow_pickle=False) as z:d={name:z[name].copy() for name in z.files}
        if 'presence' not in d:d['presence']=d['presence_logit']
        for canonical,alias in [('accepted','final_accept'),('candidate_user_id','candidate_user')]:
            if canonical not in d:d[canonical]=d[alias]
        decoded=decoder.decode(d['logits'],d['presence'],k)
        for field in ['pred_bits','accepted','candidate_user_id']:
            assert np.array_equal(d[field],decoded[field]),(version,k,field)
        positive=d['group']==f'positive_k{k}'
        d['correct']=positive&d['accepted']&(d['candidate_user_id']==d['gt_user_id'])
        d['bad']=d['accepted']&~d['correct']
        return d
    for k in [11,16]:
        old,new=load(a.v1,'V1',k),load(a.v3,'V3',k)
        for field in ['image_id','gt_bits','gt_bits_valid','gt_user_id','condition','group','scene_id']:
            assert np.array_equal(old[field],new[field]),field
        count+=len(old['logits'])+len(new['logits'])
        for condition in ['png8_control','jpeg_q90_444','jpeg_q90_420']:
            cmask=old['condition']==condition
            for group in [f'positive_k{k}','generated_negative','natural_negative']:
                mask=cmask&(old['group']==group)
                for version,d in [('V1',old),('V3',new)]:
                    n=int(mask.sum());assert n==1024
                    correct=int((mask&d['correct']).sum());bad=int((mask&d['bad']).sum());reject=int((mask&~d['accepted']).sum())
                    assert correct+bad+reject==n
                    cells.append({'k':k,'condition':condition,'group':group,'version':version,'images':n,'correct':correct,
                        'reject':reject,'false_attribution':bad,'correct_rate':correct/n,'reject_rate':reject/n,'false_attribution_rate':bad/n})
            pos=cmask&old['gt_bits_valid'];before=old['correct'][pos];after=new['correct'][pos];scene=old['scene_id'][pos]
            assert np.array_equal(np.unique(scene),np.arange(64)) and all((scene==s).sum()==16 for s in range(64))
            gains=int((~before&after).sum());losses=int((before&~after).sum())
            cluster=np.array([after[scene==s].mean()-before[scene==s].mean() for s in range(64)])
            ci=np.quantile(cluster[bootstrap].mean(axis=1),[.025,.975])
            paired.append({'k':k,'condition':condition,'images':1024,'v1_correct':int(before.sum()),'v3_correct':int(after.sum()),
                'v1_rate':float(before.mean()),'v3_rate':float(after.mean()),'new_correct_only':gains,'old_correct_only':losses,
                'net_correct':gains-losses,'delta':float((after.astype(np.int8)-before.astype(np.int8)).mean()),
                'scene_cluster_bootstrap_95_low':float(ci[0]),'scene_cluster_bootstrap_95_high':float(ci[1]),
                'scene_clusters':64,'bootstrap_draws':10000,'exact_two_sided_McNemar_p':1. if gains+losses==0 else float(binomtest(min(gains,losses),gains+losses,.5).pvalue),
                'McNemar_secondary_descriptive':True})
            if condition!='jpeg_q90_420':
                for field in ['logits','probabilities','pred_bits','accepted','candidate_user_id']:
                    identical=identical and bool(np.array_equal(old[field][cmask],new[field][cmask]))
            for version,d in [('V1',old),('V3',new)]:
                err=d['gt_bits'][pos]!=d['pred_bits'][pos];hamming=err.sum(axis=1);hist=np.bincount(hamming,minlength=33)
                x=err.astype(float);center=x-x.mean(axis=0);cov=center.T@center/len(x);std=np.sqrt(np.diag(cov));den=np.outer(std,std)
                corr=np.divide(cov,den,out=np.zeros((32,32)),where=den>0)
                errors.append({'k':k,'condition':condition,'version':version,'positive_images':1024,'BER':float(err.mean()),
                    'exact_32_bit_rate':float((hamming==0).mean()),'hamming_error_histogram':hist.tolist(),'P_E_equals_k':(hist/1024).tolist(),
                    'per_bit_error_frequency':err.mean(axis=0).tolist(),'bit_error_correlation':corr.tolist(),
                    'zero_variance_bits':np.flatnonzero(std==0).tolist(),'correlation_for_zero_variance_bits_set_to_zero':True})
    for name,value in [('paired_recovery_statistics.csv',paired),('recovery_and_false_attribution_cells.csv',cells)]:
        with (a.out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(value[0]));w.writeheader();w.writerows(value)
    (a.out/'bit_error_characterization.json').write_text(json.dumps(errors,indent=2)+'\n')
    report={'complete':True,'decisions_recomputed':count,'PNG_444_exactly_identical':identical,'paired_results':paired,
        'observed_false_attributions':sum(r['false_attribution'] for r in cells),'optimizer_steps':0,'neural_forward_views':0}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

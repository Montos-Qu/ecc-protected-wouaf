"""Descriptive paired case accounting; no training or threshold changes."""
import argparse
import json
from pathlib import Path
import numpy as np

def main(root):
    paired=[];bad_cases=[];recovery=[]
    for k in [11,16]:
        with np.load(root/f'future_test_negative_aware_k{k}.npz',allow_pickle=False) as z:new={n:z[n].copy() for n in z.files}
        for model in ['old200','candidate300']:
            with np.load(root/f'future_test_{model}_k{k}.npz',allow_pickle=False) as z:old={n:z[n].copy() for n in z.files}
            for name in ['item_id','condition','group','gt_user_id','gt_bits','gt_bits_valid']:assert np.array_equal(new[name],old[name]),name
            for condition in ['png8_control','jpeg_q90_444','jpeg_q90_420']:
                positive=(new['condition']==condition)&new['positive']
                before=old['final_accept']&old['correct_candidate'];after=new['final_accept']&new['correct_candidate']
                lost=int((positive&before&~after).sum());gained=int((positive&after&~before).sum())
                paired.append({'k':k,'reference':model,'condition':condition,'positive_images':1024,
                    'reference_correct':int((positive&before).sum()),'new_correct':int((positive&after).sum()),
                    'newly_correct':gained,'lost_correct':lost,'net_correct_difference':gained-lost,
                    'percentage_point_difference':100*(gained-lost)/1024,'paired_same_content_and_fingerprint':True,
                    'descriptive_only_no_pooled_compression_variants':True})
        for condition in ['png8_control','jpeg_q90_444','jpeg_q90_420']:
            mask=new['positive']&(new['condition']==condition)
            recovery.append({'k':k,'condition':condition,'images':int(mask.sum()),
                             'BER':float(new['bit_errors'][mask].mean()/32),
                             'P_E_k':{str(int(n)):float((new['bit_errors'][mask]==n).mean()) for n in np.unique(new['bit_errors'][mask])},
                             'correct_before_presence_filter':int((mask&new['bit_gate']&new['correct_candidate']).sum()),
                             'correct_after_presence_filter':int((mask&new['final_accept']&new['correct_candidate']).sum())})
        for i in np.flatnonzero(new['final_accept']&new['bad_candidate']):
            bad_cases.append({'k':k,'item_id':str(new['item_id'][i]),'condition':str(new['condition'][i]),'source_group':str(new['group'][i]),
                'candidate_user_id':int(new['candidate_user'][i]),'gt_user_id':int(new['gt_user_id'][i]),
                'presence_logit':float(new['presence'][i]),'candidate_loglikelihood':float(new['candidate_loglikelihood'][i]),
                'distance32':int(new['distance32'][i]),'source_zip_member':f"{new['group'][i]}/{int(new['sample_id'][i])}.png"})
    for name,data in [('paired_correct_changes.json',paired),('new_model_bit_error_and_presence_cost.json',recovery),('remaining_bad_cases.json',bad_cases)]:
        (root/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('DESCRIPTIVE_PAIRED_REVIEW_SAVED',len(paired),'comparisons;',len(bad_cases),'bad records (correlated conditions/codes)')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)

"""Real-image bit errors and filter cost; never generates or perturbs fingerprints."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
assert json.loads((root/'test_inference_verification.json').read_text())['complete']
rows=[]
for k in [11,16]:
    with np.load(root/f'future_test_negative_aware_k{k}.npz',allow_pickle=False) as z:d={n:z[n].copy() for n in z.files}
    for condition in ['png8_control','jpeg_q90_444','jpeg_q90_420']:
        mask=d['positive']&(d['condition']==condition)
        error=(d['pred_bits'][mask]!=d['gt_bits'][mask]).astype(np.float64)
        assert error.shape==(1024,32) and np.array_equal(error.sum(axis=1),d['bit_errors'][mask])
        frequency=error.mean(axis=0);centered=error-frequency
        covariance=centered.T@centered/len(error);scale=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
        correlation=np.divide(covariance,scale,out=np.full_like(covariance,np.nan),where=scale>0)
        histogram=np.bincount(error.sum(axis=1).astype(np.int64),minlength=33)
        np.savez_compressed(root/f'real_error_characterization_k{k}_{condition}.npz',sample_id=d['sample_id'][mask],item_id=d['item_id'][mask],
                            gt_bits=d['gt_bits'][mask],pred_bits=d['pred_bits'][mask],error_mask=error.astype(np.uint8),
                            per_image_hamming_errors=error.sum(axis=1).astype(np.int64),BER=frequency.mean(),P_E_k=histogram/len(error),
                            per_bit_error_frequency=frequency,bit_error_correlation=correlation)
        cells=[]
        for group in [f'positive_k{k}','generated_negative','natural_negative']:
            chosen=(d['group']==group)&(d['condition']==condition)
            cells.append({'group':group,'images':int(chosen.sum()),'bad_before_presence':int((chosen&d['bit_gate']&d['bad_candidate']).sum()),
                          'bad_after_presence':int((chosen&d['final_accept']&d['bad_candidate']).sum()),
                          'correct_before_presence':int((chosen&d['bit_gate']&d['correct_candidate']).sum()),
                          'correct_after_presence':int((chosen&d['final_accept']&d['correct_candidate']).sum())})
        rows.append({'k':k,'condition':condition,'positive_images':1024,'BER':float(frequency.mean()),
                     'P_E_k':(histogram/len(error)).tolist(),'per_bit_error_frequency':frequency.tolist(),
                     'undefined_correlations_stored_as_nan_in_NPZ_only':int(np.isnan(correlation).sum()),'presence_filter_contribution':cells,
                     'descriptive_real_image_errors_no_synthetic_simulation':True})
(root/'real_error_and_filter_characterization.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('REAL_IMAGE_CHARACTERIZATION_SAVED',6,'code/condition sets, no simulated bit flips')

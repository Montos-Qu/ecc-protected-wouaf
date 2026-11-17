"""Check that the versioned V3 delta loads in a fresh extraction of the V1 inference package."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
plan=json.loads((root/'plan.json').read_text());local=Path(plan['local'])
release=json.loads((Path(plan['parent_release_root'])/'release_inventory.json').read_text())
base_zip=Path(plan['parent_release_root'])/release['pretrained_zip']
destination=local/'portable_v3_check';destination.mkdir(exist_ok=False)
with zipfile.ZipFile(base_zip) as z:z.extractall(destination)
base=destination/'ecc-protected-wouaf';version=base/'experiments/v3';version.mkdir(parents=True)
with zipfile.ZipFile(root/'CANDIDATE_BUNDLE.zip') as z:z.extractall(version)
config=json.loads((version/'inference_config.json').read_text());assert config['version']=='v3_multiview_guard'
manifest=json.loads((base/'demo/manifest.json').read_text());demo=next(r for r in manifest if r['group']=='positive_k11')
out=base/'outputs/v3_prediction.json';out.parent.mkdir()
command=[sys.executable,str(base/'scripts/run_locked.py'),str(base/'infer.py'),'--config',str(version/'inference_config.json'),
 '--image',str(base/demo['image']),'--code','11','--condition','jpeg_q90_420','--output',str(out)]
result=subprocess.run(command,cwd='/tmp',capture_output=True,text=True,timeout=120)
assert result.returncode==0,result.stderr[-4000:]
prediction=json.loads(out.read_text())
with np.load(root/'regression_v3_k11.npz') as z:
 selected=np.flatnonzero((z['sample_id']==demo['sample_id'])&(z['group']=='positive_k11')&(z['condition']=='jpeg_q90_420'))
 assert len(selected)==1;i=int(selected[0])
 assert np.allclose(prediction['logits'],z['logits'][i],atol=1e-4,rtol=1e-4)
 assert np.array_equal(prediction['pred_bits'],z['pred_bits'][i])
 assert (prediction['decision']=='ACCEPT')==bool(z['final_accept'][i])
 if z['final_accept'][i]:assert prediction['user_id']==int(z['candidate_user'][i])
 else:assert prediction['user_id'] is None
config_sha=hashlib.sha256((version/'inference_config.json').read_bytes()).hexdigest()
prediction.update({'model_version':root.name,'model_weights_sha256':config['weights_sha256'],'inference_config_sha256':config_sha})
(root/'portable_v3_example.json').write_text(json.dumps(prediction,indent=2)+'\n')
report={'complete':True,'fresh_v1_package_extracted':True,'v3_delta_in_separate_experiments_folder':True,'new_config_explicitly_selected':True,
 'execution_cwd_independent':True,'strict_reference_runtime':prediction['runtime']['reference_runtime_verified'],
 'sample_id':demo['sample_id'],'condition':'jpeg_q90_420','logits_bits_and_decision_match_saved_v3':True,
 'selected_model_sha256':config['weights_sha256'],'inference_config_sha256':config_sha,'default_v1_files_unchanged':True}
# Every original V1 file remains byte-identical even after adding the isolated V3 folder.
for line in (base/'SHA256SUMS.txt').read_text().splitlines():
 expected,name=line.split('  ',1);assert hashlib.sha256((base/name).read_bytes()).hexdigest()==expected,name
(root/'portable_inference_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('V3_PORTABLE_INFERENCE_PASSED',json.dumps(report),flush=True)

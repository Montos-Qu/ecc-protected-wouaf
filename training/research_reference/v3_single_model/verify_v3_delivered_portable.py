"""Load each isolated V3 configuration from a fresh copy of the frozen V1 release."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
own=json.loads((root/'plan.json').read_text())
refined='parent_release_root' not in own
plan=json.loads((root.parent/'plan.json').read_text()) if refined else own
local=Path(plan['local']);release=Path(plan['parent_release_root'])
base_zip=release/'ecc-protected-wouaf-pretrained.zip'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(base_zip)=='f0bb7cbbd7b2da23cbddf9947548c2989594d93536f7fdb7f506ebd9d1eb951b'
def extract(archive,destination):
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for item in z.infolist():
   name=Path(item.filename);assert not name.is_absolute() and '..' not in name.parts
  z.extractall(destination)
destination=local/('portable_v3_refinement_check' if refined else 'portable_v3_check')
destination.mkdir(exist_ok=False);extract(base_zip,destination)
base=destination/'ecc-protected-wouaf';version=base/'experiments/v3';version.mkdir(parents=True)
extract(root/'CANDIDATE_BUNDLE.zip',version)
for line in (version/'BUNDLE_SHA256SUMS.txt').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(version/name)==expected,name
config=json.loads((version/'inference_config.json').read_text())
assert config['version']==('v3_multiview_guard_refined' if refined else 'v3_multiview_guard')
manifest=json.loads((base/'demo/manifest.json').read_text());demo=next(r for r in manifest if r['group']=='positive_k11')
out=base/'outputs/v3_prediction.json';out.parent.mkdir()
result=subprocess.run([sys.executable,str(base/'scripts/run_locked.py'),str(base/'infer.py'),'--config',str(version/'inference_config.json'),
 '--image',str(base/demo['image']),'--code','11','--condition','jpeg_q90_420','--output',str(out)],cwd='/tmp',capture_output=True,text=True,timeout=120)
assert result.returncode==0,result.stderr[-4000:]
prediction=json.loads(out.read_text())
with np.load(root/'regression_v3_k11.npz') as z:
 ids=np.flatnonzero((z['sample_id']==demo['sample_id'])&(z['group']=='positive_k11')&(z['condition']=='jpeg_q90_420'))
 assert len(ids)==1;i=int(ids[0])
 assert np.allclose(prediction['logits'],z['logits'][i],atol=1e-4,rtol=1e-4)
 assert np.array_equal(prediction['pred_bits'],z['pred_bits'][i])
 assert (prediction['decision']=='ACCEPT')==bool(z['final_accept'][i])
 assert prediction['user_id']==(int(z['candidate_user'][i]) if z['final_accept'][i] else None)
for line in (base/'SHA256SUMS.txt').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(base/name)==expected,name
prediction.update({'model_version':config['version'],'model_weights_sha256':config['weights_sha256'],'inference_config_sha256':sha(version/'inference_config.json')})
(root/'portable_v3_example.json').write_text(json.dumps(prediction,indent=2)+'\n')
report={'complete':True,'fresh_v1_package_extracted':True,'v1_zip_hash_verified':True,'all_bundle_hashes_verified':True,
 'v3_delta_in_separate_experiments_folder':True,'new_config_explicitly_selected':True,'execution_cwd_independent':True,
 'strict_reference_runtime':prediction['runtime']['reference_runtime_verified'],'sample_id':demo['sample_id'],
 'condition':'jpeg_q90_420','logits_bits_and_decision_match_saved_v3':True,'default_v1_files_unchanged':True,
 'selected_model_sha256':config['weights_sha256'],'inference_config_sha256':sha(version/'inference_config.json')}
(root/'portable_inference_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('V3_PORTABLE_INFERENCE_PASSED',json.dumps(report),flush=True)

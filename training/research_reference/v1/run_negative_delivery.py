"""Finish this experiment's audits and persistence after its active pipeline ends."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

root=Path(sys.argv[1]);local=Path(sys.argv[2]);p=json.loads((root/'plan.json').read_text())
clean=os.environ.copy();clean.pop('PYTHONPATH',None);clean.pop('PYTHONHOME',None)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:clean[name]='1'
logs=local/'delivery_logs';logs.mkdir(exist_ok=True)
complete=[]
def status(stage,state='running',error=None):
    data={'state':state,'stage':stage,'completed':complete,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if error:data['error']=error
    temp=root/'delivery_status.tmp';temp.write_text(json.dumps(data,indent=2));temp.replace(root/'delivery_status.json')
def run(stage,args,timeout=1200,save_log=True):
    status(stage)
    log=logs/(stage+'.log')
    with log.open('w') as f:result=subprocess.run(args,env=clean,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if save_log:shutil.copy2(log,root/('delivery_'+stage+'.log'))
    if result.returncode:raise RuntimeError(stage+' failed '+str(result.returncode))
    complete.append(stage)
def frozen(script,*args):return [sys.executable,p['launcher'],str(root/script),*args]

try:
    status('waiting_for_fixed_experiment')
    started=time.monotonic()
    while True:
        current=json.loads((root/'pipeline_status.json').read_text())
        if current['state']=='failed':raise RuntimeError('Research pipeline failed: '+json.dumps(current))
        if current['state']=='complete':break
        if time.monotonic()-started>10800:raise RuntimeError('Wait exceeded operational limit')
        time.sleep(10)
    locked=json.loads((root/'audit_protocol_lock.json').read_text())
    assert hashlib.sha256((root/'audit_negative_aware.py').read_bytes()).hexdigest()==locked['audit_script_sha256']
    for name,expected in json.loads((root/'descriptive_review_lock.json').read_text())['scripts'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    run('independent_audit',frozen('audit_negative_aware.py','--root',str(root),'--local',str(local)))
    run('paired_accounting',frozen('review_negative_aware_results.py','--root',str(root)),300)
    run('real_bit_characterization',frozen('characterize_negative_aware_errors.py','--root',str(root)),300)
    run('saved_checkpoint_load',frozen('verify_saved_negative_candidate.py','--root',str(root)),300)
    run('initial_package',[sys.executable,str(root/'finalize_negative_aware.py'),str(root)],1200,False)
    # Capture outside the package: its manifest must remain stable during this verification.
    run('review_restore_entry',[sys.executable,str(root/'restore_negative_aware_review.py')],1800,False)
    shutil.copy2(logs/'initial_package.log',root/'delivery_initial_package.log')
    shutil.copy2(logs/'review_restore_entry.log',root/'restore_entry_verification.log')
    (root/'restore_entry_roundtrip.json').write_text(json.dumps({'complete':True,'exact_saved_review_entry_executed':True,
        'frozen_environment_guard_and_candidate_state_load_passed':True,'existing_session_reverified':True,
        'fresh_empty_environment_not_repeated_this_run':True,'no_training_or_new_evaluation_started':True},indent=2)+'\n')
    run('final_package',[sys.executable,str(root/'finalize_negative_aware.py'),str(root)],1200,False)
    status('complete','complete')
except Exception:
    status('failed','failed',traceback.format_exc());raise

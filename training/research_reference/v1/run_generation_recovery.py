from pathlib import Path
import json,os,subprocess,sys,traceback,signal,datetime
root=Path(sys.argv[1]);local=Path(sys.argv[2]);p=json.loads((root/'plan.json').read_text())
env=os.environ.copy()
for key in ['PYTHONPATH','PYTHONHOME']:env.pop(key,None)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
completed=['prepare','train','calibrate']
def status(value):
 target=root/'pipeline_status.json';tmp=root/'pipeline_status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(target)
schedule=[('generator_backend_diagnosis',str(root/'generate_locked_negative_test.py'),'diagnose',600),('newtest_recovery',str(root/'generate_locked_negative_test.py'),'resume',5400),('test',str(root/'wouaf_negative_aware.py'),'test',2400),('report',str(root/'wouaf_negative_aware.py'),'report',300)]
try:
 for stage,script,mode,limit in schedule:
  status({'state':'running','stage':stage,'completed_stages':completed,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  with (root/(stage+'.log')).open('w') as log:
   process=subprocess.Popen([sys.executable,p['launcher'],script,mode,'--root',str(root),'--local',str(local)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise RuntimeError(stage+' operational timeout')
  if code:raise RuntimeError(stage+' failed with return code '+str(code))
  completed.append('newtest' if stage=='newtest_recovery' else stage)
 status({'state':'complete','completed_stages':completed,'operational_generator_backend_recovery':True})
except Exception:
 status({'state':'failed','stage':stage,'completed_stages':completed,'error':traceback.format_exc()});raise

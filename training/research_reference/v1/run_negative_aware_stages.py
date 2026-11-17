from pathlib import Path
import json,os,subprocess,sys,time,traceback,signal,datetime
root=Path(sys.argv[1]);local=Path(sys.argv[2]);p=json.loads((root/'plan.json').read_text())
env=os.environ.copy()
for key in ['PYTHONPATH','PYTHONHOME']:env.pop(key,None)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
complete=[]
def status(value):
 target=root/'pipeline_status.json';temporary=root/'pipeline_status.tmp'
 temporary.write_text(json.dumps(value,indent=2));temporary.replace(target)
try:
 for stage,limit in [('prepare',1800),('train',7500),('calibrate',1200),('newtest',5400),('test',2400),('report',300)]:
  status({'state':'running','stage':stage,'completed_stages':complete,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  with (root/(stage+'.log')).open('w') as log:
   process=subprocess.Popen([sys.executable,p['launcher'],str(root/'wouaf_negative_aware.py'),stage,'--root',str(root),'--local',str(local)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise RuntimeError(stage+' operational timeout')
  if code:raise RuntimeError(stage+' failed with return code '+str(code))
  complete.append(stage)
 status({'state':'complete','completed_stages':complete})
except Exception:
 status({'state':'failed','stage':stage,'completed_stages':complete,'error':traceback.format_exc()});raise

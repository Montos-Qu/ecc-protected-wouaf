"""Install portable manifests for the separately restored independent native image assets."""
import argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/independent');a=p.parse_args()
a.data.mkdir(parents=True,exist_ok=True)
for name in ['manifest.json','source_manifest.json','new_prompt_manifest.json']:
    source=ROOT/'reports/independent'/name;target=a.data/name
    if target.exists():assert target.read_bytes()==source.read_bytes(),'Existing manifest differs.'
    else:shutil.copy2(source,target)
print('INDEPENDENT_MANIFESTS_READY',a.data)

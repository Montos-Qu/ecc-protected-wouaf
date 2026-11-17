"""Explicit optional architecture initialization; not used by default inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from ecc_wouaf.core import Extractor,reference_runtime

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
reference_runtime();torch.manual_seed(20261029)
assert not a.output.exists(),'Do not overwrite a trained checkpoint.'
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.save(Extractor().state_dict(),a.output)
a.output.with_suffix('.metadata.json').write_text(json.dumps({'trained':False,'purpose':'architecture development only; not the validated fine-tuning initialization'},indent=2)+'\n')
print('UNTRAINED_ARCHITECTURE_SAVED',a.output)

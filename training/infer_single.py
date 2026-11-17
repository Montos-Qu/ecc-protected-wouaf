"""Explicit opt-in inference for a separately trained single extractor configuration."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecc_wouaf.core import Predictor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
p.add_argument('--image',type=Path,required=True);p.add_argument('--code',type=int,choices=[11,16],default=11)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=Predictor(a.config).predict(a.image,a.code)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Load the trained final V3 by default; V1/V2 are explicit comparisons."""
import argparse,json
from pathlib import Path
from ecc_wouaf import Predictor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--code',type=int,choices=[11,16],default=11)
    p.add_argument('--version',choices=['v3','v1','v2'],default='v3')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--allow-unvalidated-runtime',action='store_true')
    a=p.parse_args();model=Predictor(a.version,a.device,not a.allow_unvalidated_runtime)
    result=model.predict(a.image,a.code);text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)
if __name__=='__main__':main()

"""Load the trained final V3 by default; V1/V2 are explicit comparisons."""
import argparse,json
from pathlib import Path
from ecc_wouaf import Predictor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--code',type=int,choices=[11,16],default=11)
    p.add_argument('--version',choices=['v3','v1','v2'],default='v3')
    p.add_argument('--resize',choices=['native','fit'],default='native')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--allow-unvalidated-runtime',action='store_true')
    a=p.parse_args();model=Predictor(a.version,a.device,not a.allow_unvalidated_runtime)
    if a.resize=='native':result=model.predict(a.image,a.code)
    else:
        from image_preprocessing import predict_encoded
        result=predict_encoded(model,a.image.read_bytes(),a.code,a.image.name,a.resize)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__':main()

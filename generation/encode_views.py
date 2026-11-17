"""Explicitly create real PNG/JPEG Q90 444/420 views from a labeled PNG source set."""
import argparse,io,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image,JpegImagePlugin
from ecc_wouaf.core import sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.source/'manifest.json').read_text());a.out.mkdir(parents=True,exist_ok=False);result=[]
for row in rows:
    source=(a.source/row['image']).resolve();assert source.is_relative_to(a.source.resolve()) and sha256(source)==row['sha256']
    with Image.open(source) as f:
        assert f.format=='PNG' and f.size==(512,512),'This experimental encoder expects native 512x512 PNG sources.'
        image=f.convert('RGB').copy()
    for condition,sampling in [('png8_control',None),('jpeg_q90_444',0),('jpeg_q90_420',2)]:
        relative=Path(condition)/row['group']/(str(row['sample_id'])+('.png' if sampling is None else '.jpg'))
        path=a.out/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if sampling is None:shutil.copy2(source,path)
        else:
            image.save(path,format='JPEG',quality=90,subsampling=sampling,optimize=False,progressive=False)
            with Image.open(path) as f:assert JpegImagePlugin.get_sampling(f)==sampling
        result.append({**row,'image_id':f"{row['group']}:{row['sample_id']}:{condition}",'image':relative.as_posix(),
            'sha256':sha256(path),'condition':condition,'container':'PNG' if sampling is None else 'JPEG','jpeg_sampling':sampling})
(a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('REAL_ENCODED_VIEWS_SAVED',len(result))

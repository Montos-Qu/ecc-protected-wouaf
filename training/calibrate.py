"""Fit only the held-out calibration role, after checkpoint selection is locked."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from ecc_wouaf.core import reference_runtime,Extractor,Decoder,load_config,sha256
from recipe import dataset,image_cache,infer,decision_data,rule,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    reference_runtime();lock=json.loads((a.run/'selection_locked.json').read_text())
    assert lock['complete'] and sha256(a.run/'selected.pth')==lock['weights_sha256']
    assert not (a.run/'inference_config.json').exists(),'Calibration is already locked.'
    records,targets=dataset(a.data);cache=image_cache(a.data,records,a.run/'pixels.uint8')
    model=Extractor().eval().float().cuda();model.load_state_dict(torch.load(a.run/'selected.pth',map_location='cpu'),strict=True)
    config=load_config();decoder=Decoder(config);ds=decision_data(records,infer(model,records,cache,'calibration'),decoder)
    result={}
    for k,d in ds.items():
        threshold,measured=rule(d);config['policies'][str(k)]['presence_logit_min']=threshold;result[k]={'threshold':threshold,'metrics':measured}
    config.pop('_configuration_directory',None);config['weights']='selected.pth';config['weights_relative_to']='configuration_directory'
    config['weights_sha256']=lock['weights_sha256'];config['trained']=True
    save(a.run/'calibration_metrics.json',result);save(a.run/'inference_config.json',config)
    print('CALIBRATION_LOCKED',a.run/'inference_config.json')
if __name__=='__main__':main()

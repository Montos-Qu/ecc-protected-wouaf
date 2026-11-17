"""Default trained V3 system; routing follows the actual encoded file header."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, JpegImagePlugin
from .core import ROOT, Predictor as SingleExtractor, load_config

def read_encoded(data):
    with Image.open(io.BytesIO(data)) as image:
        container=image.format
        sampling=JpegImagePlugin.get_sampling(image) if container=='JPEG' else None
        pixels=np.asarray(image.convert('RGB')).copy()
    if pixels.shape!=(512,512,3):
        raise ValueError(f'Expected 512x512 RGB input; received {pixels.shape}. No resizing is applied.')
    branch='jpeg420_expert' if container=='JPEG' and sampling==2 else 'clean_v1'
    return pixels,{'container':container,'jpeg_sampling':sampling,'selected_branch':branch,
        'route_source':'actual_file_header','filename_used_for_routing':False,'additional_JPEG_reencoding':False,
        'validated_container_sampling':container=='PNG' or (container=='JPEG' and sampling in (0,2))}

class Predictor:
    """Loads trained V3 by default. Use version='v1'/'v2' for fixed-extractor controls."""
    def __init__(self,version='v3',device='cuda',strict=True):
        if version not in ('v1','v2','v3'):raise ValueError('version must be v1, v2, or v3')
        self.version=version;self.device=device;self.strict=strict;self.models={}
        self.system=json.loads((ROOT/'configs/pretrained.json').read_text())
        assert self.system['version']=='v3_format_aware_v1_v2' and self.system['trained'] is True
        self.paths={branch:ROOT/path for branch,path in self.system['branches'].items()}
        configs={branch:load_config(path) for branch,path in self.paths.items()}
        assert configs['clean_v1']['policies']==configs['jpeg420_expert']['policies'],'V3 policies must remain identical.'
        for branch,config in configs.items():
            assert config['weights_sha256']==self.system['weights_sha256'][branch],'Unexpected V3 component.'
    def model(self,branch):
        if branch not in self.models:
            self.models[branch]=SingleExtractor(self.paths[branch],self.device,self.strict)
        return self.models[branch]
    def predict_encoded(self,data,k=11,image_id='encoded_image'):
        if k not in (11,16):raise ValueError('Supported user capacities use k=11 or k=16.')
        if not isinstance(data,bytes):raise TypeError('Pass original encoded bytes, not an RGB tensor.')
        pixels,metadata=read_encoded(data)
        if self.version!='v3':
            metadata['selected_branch']='clean_v1' if self.version=='v1' else 'jpeg420_expert'
            metadata['route_source']='explicit_fixed_extractor_comparison'
        model=self.model(metadata['selected_branch'])
        raw=model.predict_batch(pixels[None],k);accepted=bool(raw['accepted'][0])
        reason=None
        if not raw['ecc_accept'][0]:reason='outside_BCH_decoding_radius'
        elif not raw['bit_gate'][0]:reason='bit_confidence_below_threshold'
        elif not accepted:reason='watermark_presence_below_threshold'
        return {'image_id':str(image_id),'input_sha256':hashlib.sha256(data).hexdigest(),
            'model_version':self.system['version'] if self.version=='v3' else self.version,
            'requested_version':self.version,**metadata,'model_weights_sha256':model.config['weights_sha256'],
            'decision':'ACCEPT' if accepted else 'REJECT','user_id':int(raw['candidate_user_id'][0]) if accepted else None,
            'reject_reason':reason,'code':f'BCH(31,{k})+P','capacity':2**k,
            'logits':raw['logits'][0].tolist(),'probabilities':raw['probabilities'][0].tolist(),
            'pred_bits':raw['pred_bits'][0].tolist(),'presence_logit':float(raw['presence_logit'][0]),
            'distance32':int(raw['distance32'][0]),'candidate_loglikelihood':float(raw['candidate_loglikelihood'][0]),
            'policy':model.config['policies'][str(k)],'runtime':model.runtime,
            'model_status':'trained_final_research_system' if self.version=='v3' else 'trained_comparison_extractor'}
    def predict(self,image,k=11):
        path=Path(image)
        return self.predict_encoded(path.read_bytes(),k,path.name)

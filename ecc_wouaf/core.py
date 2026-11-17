import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np
from PIL import Image, JpegImagePlugin
import torch
from torch import nn
from torchvision.models import resnet50

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=['png8_control','jpeg_q90_444','jpeg_q90_420']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def reference_runtime(strict=True,device='cuda'):
    profile=json.loads((ROOT/'environment/locked/locked_environment_profile.json').read_text())
    if strict:
        assert device=='cuda','Reference mode requires CUDA; CPU is an explicit unvalidated mode.'
        assert Path(sys.prefix).resolve()==Path(profile['prefix']).resolve(),'Restore and use scripts/run_locked.py.'
        assert platform.python_version()==profile['python'],'Python version differs from the frozen reference.'
        normalize=lambda name:re.sub(r'[-_.]+','-',name).lower()
        installed={normalize(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}
        assert installed==profile['python_distributions'],'Package inventory drift; do not upgrade the frozen environment.'
        native={r['name']:[r['version'],r['build']] for r in [json.loads(p.read_text()) for p in (Path(sys.prefix)/'conda-meta').glob('*.json')]}
        assert native==profile['native_builds'],'Native build inventory drift.'
        result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
        assert result.stdout.strip(),'No reference GPU found.'
        for line in result.stdout.splitlines():
            name,driver=line.split(',',1)
            assert name.strip()==profile['required_gpu'] and driver.strip()==profile['required_driver'],'GPU/driver differs from reference.'
        assert not os.environ.get('PYTHONPATH'),'Use the isolated launcher.'
    if device=='cuda':assert torch.cuda.is_available(),'CUDA is unavailable.'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return {'reference_runtime_verified':strict,'device':device,'python':platform.python_version(),'torch':torch.__version__}

class Extractor(nn.Module):
    """Construction is uninitialized; Predictor always verifies and loads trained weights."""
    def __init__(self):
        super().__init__()
        self.backbone=resnet50(weights=None)
        self.backbone.fc=nn.Identity()
        self.bit_head=nn.Linear(2048,32)
        self.presence_head=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,1))
    def forward(self,x):
        features=self.backbone(x)
        return self.bit_head(features),self.presence_head(features)[:,0]

def image_pixels(path,condition='png8_control'):
    assert condition in CONDITIONS
    with Image.open(path) as f:image=f.convert('RGB').copy()
    if image.size!=(512,512):
        raise ValueError(f'Expected 512x512, received {image.size}. No silent resizing is performed.')
    if condition!='png8_control':
        buf=io.BytesIO();sampling=0 if condition.endswith('444') else 2
        image.save(buf,format='JPEG',quality=90,subsampling=sampling,optimize=False,progressive=False);buf.seek(0)
        with Image.open(buf) as f:
            assert JpegImagePlugin.get_sampling(f)==sampling
            image=f.convert('RGB').copy()
    return np.asarray(image).copy()

def normalize_pixels(batch,device):
    values=np.asarray(batch)
    assert values.dtype==np.uint8 and values.ndim==4 and values.shape[1:]==(512,512,3)
    x=torch.from_numpy(values.astype(np.float32)/255).permute(0,3,1,2).contiguous().to(device)
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    return (x-mean)/std

class Decoder:
    def __init__(self,config):
        self.config=config;self.books={}
        for k in [11,16]:
            directory=ROOT/'ecc'/f'bch31_{k}_p'
            policy=json.loads((directory/'decoder_policy.json').read_text())
            with np.load(directory/'bounded_distance_syndrome_table.npz') as z:table={name:z[name].copy() for name in z.files}
            generator=np.load(directory/'generator_matrix.npz')['generator']
            self.books[k]=(policy,table,generator)
    def decode(self,logits,presence,k=11):
        assert k in self.books
        policy,table,generator=self.books[k]
        assert logits.ndim==2 and logits.shape[1]==32 and len(presence)==len(logits)
        assert np.isfinite(logits).all() and np.isfinite(presence).all()
        probabilities=torch.from_numpy(np.asarray(logits)).sigmoid().numpy()
        received=(probabilities>.5).astype(np.uint8)
        words=np.sum(received[:,:31].astype(np.uint64)*(np.uint64(1)<<np.arange(30,-1,-1,dtype=np.uint64)),axis=1).astype(np.uint32)
        syndrome=words.copy();degree=31-k
        for exponent in range(30,degree-1,-1):
            chosen=(syndrome&np.uint32(1<<exponent))!=0
            syndrome[chosen]^=np.uint32(policy['generator_polynomial_integer']<<(exponent-degree))
        positions=np.searchsorted(table['syndrome'],syndrome);safe=np.minimum(positions,len(table['syndrome'])-1)
        valid=(positions<len(table['syndrome']))&(table['syndrome'][safe]==syndrome)
        corrected=words^np.where(valid,table['error_word'][safe],0).astype(np.uint32)
        word31=((corrected[:,None]>>np.arange(30,-1,-1))&1).astype(np.uint8)
        messages=word31[:,:k]
        assert np.array_equal(messages[valid]@generator%2,word31[valid])
        word32=np.column_stack([word31,word31.sum(axis=1)%2])
        distance=(word32!=received).sum(axis=1)
        ecc=valid&(distance<=policy['t'])
        score=-np.logaddexp(0,-(2*word32.astype(float)-1)*logits.astype(float)).mean(axis=1)
        thresholds=self.config['policies'][str(k)]
        bit_gate=ecc&(score>=thresholds['bit_loglikelihood_min'])
        accepted=bit_gate&(presence>=thresholds['presence_logit_min'])
        users=np.sum(messages.astype(np.uint64)*(np.uint64(1)<<np.arange(k-1,-1,-1,dtype=np.uint64)),axis=1).astype(np.int64)
        return {'accepted':accepted,'candidate_user_id':users,'ecc_accept':ecc,'bit_gate':bit_gate,
                'distance32':distance,'candidate_loglikelihood':score,'pred_bits':received,'probabilities':probabilities}

def load_config(path=None):
    # The public package Predictor defaults to the V3 system in predictor.py.
    path=Path(path).resolve() if path else ROOT/'configs/versions/v1.json'
    config=json.loads(path.read_text())
    assert config['trained'] is True,'Inference refuses an untrained model configuration.'
    for relative,value in config['asset_sha256'].items():
        assert sha256(ROOT/relative)==value,f'Asset mismatch: {relative}'
    config['_configuration_directory']=str(path.parent)
    return config

class Predictor:
    def __init__(self,config_path=None,device='cuda',strict=True):
        self.runtime=reference_runtime(strict,device);self.device=device
        self.config=load_config(config_path)
        model_path=Path(self.config['weights'])
        if not model_path.is_absolute():
            base=Path(self.config['_configuration_directory']) if self.config.get('weights_relative_to')=='configuration_directory' else ROOT
            model_path=base/model_path
        if model_path.is_file() and model_path.stat().st_size<1024:
            if model_path.read_bytes().startswith(b'version https://git-lfs.github.com/spec/'):
                raise RuntimeError('Weights are a Git LFS pointer. Run git lfs pull or extract the pretrained release ZIP.')
        assert sha256(model_path)==self.config['weights_sha256'],'Trained checkpoint checksum mismatch.'
        self.model=Extractor()
        self.model.load_state_dict(torch.load(model_path,map_location='cpu'),strict=True)
        self.model.requires_grad_(False).eval().float().to(device)
        self.decoder=Decoder(self.config)
    def predict_batch(self,batch,k=11):
        with torch.no_grad():bits,presence=self.model(normalize_pixels(batch,self.device))
        logits=bits.cpu().numpy();presence=presence.cpu().numpy()
        decoded=self.decoder.decode(logits,presence,k)
        return {**decoded,'logits':logits,'presence_logit':presence}
    def predict(self,image,k=11,condition='png8_control'):
        result=self.predict_batch(image_pixels(image,condition)[None],k)
        accepted=bool(result['accepted'][0]);policy=self.config['policies'][str(k)]
        reason=None
        if not result['ecc_accept'][0]:reason='outside_BCH_decoding_radius'
        elif not result['bit_gate'][0]:reason='bit_confidence_below_threshold'
        elif not accepted:reason='watermark_presence_below_threshold'
        return {'image_id':Path(image).name,'decision':'ACCEPT' if accepted else 'REJECT',
                'user_id':int(result['candidate_user_id'][0]) if accepted else None,'reject_reason':reason,
                'code':f'BCH(31,{k})+P','capacity':2**k,'condition':condition,
                'additional_JPEG_reencoding':condition!='png8_control',
                'logits':result['logits'][0].tolist(),'probabilities':result['probabilities'][0].tolist(),
                'pred_bits':result['pred_bits'][0].tolist(),'presence_logit':float(result['presence_logit'][0]),
                'distance32':int(result['distance32'][0]),'candidate_loglikelihood':float(result['candidate_loglikelihood'][0]),
                'policy':policy,'runtime':self.runtime,'model_status':'trained_research_candidate'}

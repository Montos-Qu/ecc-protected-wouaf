"""Render the unchanged baseline-3000 from saved latent tensors and encoded users."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from ecc_wouaf.core import reference_runtime,sha256,Decoder,load_config

def vendor():
    target=ROOT/'generation/vendor'
    if not (target/'customization.py').exists():
        with zipfile.ZipFile(ROOT/'environment/locked/baseline_inference_sources.zip') as z:
            for member in z.infolist():
                path=(target/member.filename).resolve();assert path.is_relative_to(target.resolve())
                if member.is_dir():path.mkdir(parents=True,exist_ok=True)
                else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(member))
    sys.path.insert(0,str(target))

def check_snapshot(root,path):
    relative=path.relative_to(root).as_posix();matches=[]
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        parts=line.split(maxsplit=1)
        if len(parts)==2:
            name=parts[1].strip().lstrip('*').replace('\\','/')
            if name==relative or name.endswith('/'+relative):matches.append(parts[0])
    assert len(matches)==1 and sha256(path)==matches[0],relative

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--latents',type=Path,required=True)
    p.add_argument('--users',type=Path);p.add_argument('--code',type=int,choices=[11,16],default=11)
    p.add_argument('--unwatermarked',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int);a=p.parse_args();reference_runtime();torch.backends.cudnn.deterministic=False
    assert (a.users is None)==a.unwatermarked,'Provide --users for watermarked images, or --unwatermarked.'
    a.out.mkdir(parents=True,exist_ok=False);vendor()
    from attribution import MappingNetwork
    from customization import customize_vae_decoder
    assets=ROOT/'generation/assets';latent=np.load(a.latents,mmap_mode='r');assert latent.dtype==np.float32 and latent.shape[1:]==(4,64,64)
    n=len(latent) if a.limit is None else min(a.limit,len(latent));assert n>0
    if a.unwatermarked:
        pristine=assets/'pristine_vae';vae=AutoencoderKL.from_config(str(pristine/'config.json'))
        vae.load_state_dict(torch.load(pristine/'diffusion_pytorch_model.bin',map_location='cpu'),strict=True)
        mapping=None;bits=None;ids=10000000+np.arange(len(latent));users=np.full(len(latent),-1);group='generated_negative'
    else:
        with np.load(a.users,allow_pickle=False) as z:bits=z['gt_bits'].copy();users=z['user_id'].copy();ids=z['sample_id'].copy()
        assert len(bits)==len(latent) and bits.shape[1]==32
        generator=Decoder(load_config()).books[a.code][2]
        messages=((users[:,None].astype(np.uint64)>>np.arange(a.code-1,-1,-1,dtype=np.uint64))&1).astype(np.uint8)
        words=messages@generator%2;assert np.array_equal(np.column_stack([words,words.sum(axis=1)%2]),bits)
        weight=assets/'checkpoint-3000/pytorch_model.bin';mapping_file=assets/'weights/mapping_network.pth'
        check_snapshot(assets,weight);check_snapshot(assets,mapping_file)
        vae=customize_vae_decoder(AutoencoderKL.from_config(str(assets/'final_pipeline/vae/config.json')),128,1)
        vae.load_state_dict(torch.load(weight,map_location='cpu'),strict=True)
        mapping=MappingNetwork(32,128,num_layers=2);mapping.load_state_dict(torch.load(mapping_file,map_location='cpu'),strict=True)
        mapping.requires_grad_(False).eval().float().cuda();group=f'positive_k{a.code}'
    vae.requires_grad_(False).eval().float().cuda();rows=[]
    with torch.no_grad():
        for start in range(0,n,4):
            values=torch.from_numpy(np.array(latent[start:min(start+4,n)],copy=True)).float().cuda()
            image=vae.decode(values/.18215).sample if mapping is None else vae.decode(values/.18215,mapping(torch.from_numpy(bits[start:start+len(values)].copy()).float().cuda()+1e-8)).sample
            pixels=((image.clamp(-1,1)/2+.5).clamp(0,1)*255+.5).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
            for j,pixel in enumerate(pixels):
                i=start+j;path=a.out/f'{int(ids[i])}.png';Image.fromarray(pixel).save(path)
                rows.append({'sample_id':int(ids[i]),'group':group,'user_id':int(users[i]),'gt_bits':None if bits is None else bits[i].tolist(),
                    'image':path.name,'sha256':sha256(path),'latent_tensor_sha256':hashlib.sha256(np.asarray(latent[i]).tobytes()).hexdigest()})
            print('BASELINE_RENDERED',len(rows),n,flush=True)
    (a.out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()

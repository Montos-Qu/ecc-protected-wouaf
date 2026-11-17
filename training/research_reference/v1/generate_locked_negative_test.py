"""Diagnose/recover generator backend parity; retain locked data/model/gate."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np

def main(root,local,mode):
    s=importlib.util.spec_from_file_location('locked_negative_training',root/'wouaf_negative_aware.py')
    impl=importlib.util.module_from_spec(s);s.loader.exec_module(impl)
    p=impl.context(root);impl.assert_data(root);torch=impl.configure(False)
    from PIL import Image
    from torchvision import transforms
    from diffusers import AutoencoderKL
    sys.path.insert(0,p['repo']);from attribution import MappingNetwork
    from customization import customize_vae_decoder
    assert json.loads((root/'calibration_verification.json').read_text())['complete']
    policy_hash=impl.sha(root/'locked_presence_policy.json')
    assert not (root/'future_test_generation.json').exists()
    old_latents=json.loads((Path(p['fixed_root'])/'latent_manifest.json').read_text())
    for i in range(1280,1284):
        expected=next(r['sha256'] for r in old_latents if r['sample_id']==i)
        assert impl.sha(local/'future_latents'/f'{i}.pth')==expected
    for r in json.loads((root/'future_latent_manifest.json').read_text()):
        assert impl.sha(local/'future_latents'/f"{r['sample_id']}.pth")==r['sha256']
    rows=json.loads((Path(p['fixed_root'])/'image_manifest.json').read_text())
    archive=zipfile.ZipFile(Path(p['previous_local'])/'images_1.zip')
    def generator(group):
        if group=='generated_negative':
            vae=AutoencoderKL.from_config(str(Path(p['pristine_vae'])/'config.json'))
            state=torch.load(Path(p['pristine_vae'])/'diffusion_pytorch_model.bin',map_location='cpu')
            assert not any('affine' in n for n in state);vae.load_state_dict(state,strict=True);mapping=None
        else:
            vae=customize_vae_decoder(AutoencoderKL.from_config(str(Path(p['snapshot'])/'final_pipeline/vae/config.json')),128,1)
            vae.load_state_dict(torch.load(Path(p['snapshot'])/'checkpoint-3000/pytorch_model.bin',map_location='cpu'),strict=True)
            mapping=MappingNetwork(32,128,num_layers=2);mapping.load_state_dict(torch.load(Path(p['snapshot'])/'weights/mapping_network.pth',map_location='cpu'),strict=True)
            mapping.requires_grad_(False).eval().float().cuda()
        vae.requires_grad_(False).eval().float().cuda()
        return vae,mapping
    def decode(vae,mapping,ids,bits):
        latents=torch.stack([torch.load(local/'future_latents'/f'{int(i)}.pth',map_location='cpu') for i in ids]).float().cuda()
        value=vae.decode(latents/.18215).sample if bits is None else vae.decode(latents/.18215,mapping(torch.from_numpy(bits.copy()).float().cuda()+1e-8)).sample
        value=(value.clamp(-1,1)/2+.5).clamp(0,1)
        return (value*255+.5).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
    diagnostics=[];image_rows=[];started=time.perf_counter()
    if mode=='resume':
        diagnostics=json.loads((root/'generator_backend_diagnostic.json').read_text())
        assert all(r['pixel_exact'] for r in diagnostics if r['cudnn_deterministic'] is False)
        assert sum(not r['cudnn_deterministic'] for r in diagnostics)==3
        # Recover already downloaded, fixed-ID photos. No replacement/resampling.
        photo_plan=json.loads((root/'future_test_photo_selection.json').read_text());old_hashes=set(photo_plan['prior_photo_hashes'])
        resize=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512)])
        for r in photo_plan['records']:
            original=local/'future_photo_originals'/r['file_name'];assert original.is_file()
            original_sha=impl.sha(original);assert original_sha not in old_hashes
            path=local/'future_pngs/natural_negative'/f"{int(r['id'])}.png"
            with Image.open(original) as f:pixels=np.asarray(resize(f.convert('RGB')))
            with Image.open(path) as f:assert np.array_equal(pixels,np.asarray(f.convert('RGB')))
            image_rows.append({'group':'natural_negative','sample_id':int(r['id']),'user_id':-1,'gt_bits':None,
                'path':str(path),'sha256':impl.sha(path),'original_sha256':original_sha,
                'url':'http://images.cocodataset.org/test2014/'+r['file_name']})
        assert len(image_rows)==1024 and len({r['original_sha256'] for r in image_rows})==1024
        print('FIXED_FUTURE_PHOTOS_RECOVERED1024_NO_NEW_DOWNLOADS',flush=True)
    for group in ['positive_k11','positive_k16','generated_negative']:
        vae,mapping=generator(group)
        reference_group='negative' if group=='generated_negative' else group
        reference_bits=None if mapping is None else np.load(Path(p['fixed_root'])/'inputs'/f'{reference_group}.npz')['gt_bits'][:4].copy()
        reference=[]
        for i in range(1280,1284):
            r=next(r for r in rows if r['group']==reference_group and r['sample_id']==i)
            payload=archive.read(r['relative']);assert hashlib.sha256(payload).hexdigest()==r['sha256']
            with Image.open(io.BytesIO(payload)) as f:reference.append(np.asarray(f.convert('RGB')))
        reference=np.stack(reference)
        for flag in ([True,False] if mode=='diagnose' else [False]):
            torch.backends.cudnn.deterministic=flag
            values=decode(vae,mapping,range(1280,1284),reference_bits)
            delta=np.abs(values.astype(np.int16)-reference.astype(np.int16))
            record={'group':group,'cudnn_deterministic':flag,'pixel_exact':bool(np.array_equal(values,reference)),
                    'different_channel_values':int((delta!=0).sum()),'maximum_uint8_difference':int(delta.max())}
            print('GENERATOR_BACKEND_REFERENCE',json.dumps(record),flush=True)
            if mode=='diagnose':diagnostics.append(record)
            else:assert record['pixel_exact'],record
        if mode=='resume':
            inputs=None if mapping is None else np.load(root/f"future_test_inputs_k{int(group.removeprefix('positive_k'))}.npz")
            directory=local/'future_pngs'/group;directory.mkdir(parents=True,exist_ok=True)
            for start in range(0,1024,4):
                ids=np.arange(2420,3444)[start:start+4];bits=None if inputs is None else inputs['gt_bits'][start:start+4]
                values=decode(vae,mapping,ids,bits)
                for j,i in enumerate(ids):
                    path=directory/f'{int(i)}.png'
                    if path.exists():
                        with Image.open(path) as f:assert np.array_equal(values[j],np.asarray(f.convert('RGB')))
                    else:Image.fromarray(values[j]).save(path)
                    image_rows.append({'group':group,'sample_id':int(i),'user_id':-1 if inputs is None else int(inputs['user_id'][start+j]),
                        'gt_bits':None if inputs is None else bits[j].tolist(),'path':str(path),'sha256':impl.sha(path)})
                if (start+4)%128==0:print('FUTURE_GENERATED_IMAGES',group,start+4,1024,'elapsed',round(time.perf_counter()-started,1),flush=True)
        del vae,mapping;torch.cuda.empty_cache()
    archive.close()
    if mode=='diagnose':
        impl.save(root/'generator_backend_diagnostic.json',diagnostics)
    else:
        assert len(image_rows)==4096
        impl.save(root/'future_test_image_manifest.json',image_rows)
        archive_path=local/'future_test_source_images.zip'
        with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_STORED) as z:
            for r in image_rows:z.write(r['path'],arcname=f"{r['group']}/{r['sample_id']}.png")
        shutil.copy2(archive_path,root/archive_path.name);assert impl.sha(archive_path)==impl.sha(root/archive_path.name)
        impl.save(root/'future_test_generation.json',{'complete':True,'source_images':4096,'per_group':1024,
            'archived_sha256':impl.sha(root/archive_path.name),'generator_control_pixels_exact':12,
            'new_users_disjoint_from_4456_development_pool':True,'fresh_natural_photos':1024,
            'latent_content_previously_in_original_baseline5000':True,'new_branch_did_not_train_or_select_on_test_content':True,
            'policy_hash_before_generation':policy_hash,'cudnn_deterministic':False,
            'generator_backend_matches_original_baseline_generation':True,'implementation_recovery_script':str(Path(__file__)),
            'implementation_recovery_script_sha256':impl.sha(Path(__file__)),'fixed_photos_reused_without_resampling':True,
            'wall_seconds':time.perf_counter()-started})
        assert impl.sha(root/'locked_presence_policy.json')==policy_hash
        impl.context(root);impl.assert_data(root)
        print('LOCKED_FUTURE_TEST_GENERATION_RECOVERED4096',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['diagnose','resume']);p.add_argument('--root',type=Path,required=True);p.add_argument('--local',type=Path,required=True)
    a=p.parse_args();main(a.root,a.local,a.mode)

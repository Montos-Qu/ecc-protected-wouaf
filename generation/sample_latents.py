"""Sample saved prompt/seed units with frozen SD2 UNet, PNDM50, CFG7.5."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from diffusers import UNet2DConditionModel,PNDMScheduler
from transformers import CLIPTextModel,CLIPTokenizer
from ecc_wouaf.core import reference_runtime
from from_latents import check_snapshot

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    reference_runtime();torch.backends.cudnn.deterministic=False
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'generation/assets';pipeline=assets/'final_pipeline'
    for folder in ['unet','text_encoder','tokenizer','scheduler']:
        for path in (pipeline/folder).iterdir():
            if path.is_file():check_snapshot(assets,path)
    rows=json.loads(a.prompts.read_text());rows=rows if a.limit is None else rows[:a.limit];assert len(rows)>0
    unet=UNet2DConditionModel.from_pretrained(pipeline/'unet',local_files_only=True,torch_dtype=torch.float32).requires_grad_(False).eval().cuda()
    encoder=CLIPTextModel.from_pretrained(pipeline/'text_encoder',local_files_only=True,torch_dtype=torch.float32).requires_grad_(False).eval().cuda()
    tokenizer=CLIPTokenizer.from_pretrained(pipeline/'tokenizer',local_files_only=True)
    scheduler=PNDMScheduler.from_pretrained(pipeline/'scheduler',local_files_only=True);assert scheduler.config.prediction_type=='epsilon'
    def embed(text):
        tokens=tokenizer(text,padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
        attention=tokens.attention_mask.cuda() if getattr(encoder.config,'use_attention_mask',False) else None
        return encoder(tokens.input_ids.cuda(),attention_mask=attention)[0]
    output=np.lib.format.open_memmap(a.output,mode='w+',dtype=np.float32,shape=(len(rows),4,64,64))
    with torch.no_grad():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];n=len(batch)
            noise=torch.stack([torch.randn((4,64,64),generator=torch.Generator(device='cpu').manual_seed(r['initial_noise_seed']),dtype=torch.float32) for r in batch]).cuda()
            embeddings=torch.cat([embed(['']*n),embed([r['prompt'] for r in batch])])
            scheduler.set_timesteps(50,device='cuda');value=noise*scheduler.init_noise_sigma
            for t in scheduler.timesteps:
                inputs=scheduler.scale_model_input(torch.cat([value,value]),t)
                unconditional,conditional=unet(inputs,t,encoder_hidden_states=embeddings).sample.chunk(2)
                value=scheduler.step(unconditional+7.5*(conditional-unconditional),t,value).prev_sample
            output[start:start+n]=value.float().cpu().numpy();output.flush();print('NEW_LATENTS',start+n,len(rows),flush=True)
if __name__=='__main__':main()

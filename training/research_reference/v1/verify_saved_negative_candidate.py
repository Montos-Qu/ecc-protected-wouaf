import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);root=parser.parse_args().root
spec=importlib.util.spec_from_file_location('saved_negative_aware',root/'wouaf_negative_aware.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
p=module.context(root);module.assert_data(root);module.configure(False)
policy=json.loads((root/'locked_presence_policy.json').read_text())
assert module.sha(root/'selected_multitask.pth')==policy['selected_weights_sha256']
model=module.make_model(p)
model.load_state_dict(torch.load(root/'selected_multitask.pth',map_location='cpu'),strict=True)
model.eval().requires_grad_(False)
assert all(torch.isfinite(v).all() for v in model.state_dict().values())
print('SAVED_NEGATIVE_AWARE_CHECKPOINT_LOAD_VERIFIED',sum(v.numel() for v in model.parameters()),'parameters; no active configuration change')

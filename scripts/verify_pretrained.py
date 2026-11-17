"""Compare relocated trained V3 and V1 against archived real-file reference outputs."""
import argparse,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from ecc_wouaf import Predictor
from ecc_wouaf.core import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    golden=json.loads((ROOT/'demo/golden_predictions.json').read_text())
    systems={v:Predictor(v) for v in ['v1','v3','v2']};views=0;decisions=0;maximum=0.;v2_controls=0
    for row in golden:
        path=ROOT/row['image'];assert sha256(path)==row['sha256']
        for version in ['v1','v3']:
            expected=row['reference'][version];code=int(next(iter(expected['decisions'])))
            actual=systems[version].predict(path,code);bits=np.asarray(actual['logits'],np.float32)
            assert np.allclose(bits,expected['logits'],atol=1e-4,rtol=1e-4),(row['image'],version,'logits')
            assert np.isclose(actual['presence_logit'],expected['presence_logit'],atol=1e-4,rtol=1e-4)
            assert actual['pred_bits']==expected['pred_bits'] and actual['selected_branch']==expected['selected_branch']
            maximum=max(maximum,float(np.abs(bits-np.asarray(expected['logits'])).max()));views+=1
            model=systems[version].model(actual['selected_branch'])
            for key,value in expected['decisions'].items():
                d=model.decoder.decode(bits[None],np.asarray([actual['presence_logit']],np.float32),int(key))
                assert bool(d['accepted'][0])==value['accepted'] and int(d['candidate_user_id'][0])==value['candidate_user_id']
                assert int(d['distance32'][0])==value['distance32'];decisions+=1
            if actual['decision']=='REJECT':assert actual['user_id'] is None
        if row['condition']=='jpeg_q90_420':
            fixed=systems['v2'].predict(path,code);routed=systems['v3'].predict(path,code)
            for field in ['logits','probabilities','pred_bits','decision','user_id']:
                assert fixed[field]==routed[field],('V2 expert equivalence',field)
            v2_controls+=1
    example=next(r for r in golden if r['condition']=='jpeg_q90_420' and r['group']=='positive_k11')
    data=(ROOT/example['image']).read_bytes()
    misleading=systems['v3'].predict_encoded(data,11,'misleading_extension.png')
    assert misleading['selected_branch']=='jpeg420_expert' and misleading['container']=='JPEG'
    with Image.open(io.BytesIO(data)) as f:
        stream=io.BytesIO();f.convert('RGB').save(stream,format='PNG')
    wrapped=systems['v3'].predict_encoded(stream.getvalue(),11,'converted.png')
    assert wrapped['selected_branch']=='clean_v1' and wrapped['container']=='PNG'
    small=io.BytesIO();Image.new('RGB',(16,16)).save(small,format='PNG')
    try:systems['v3'].predict_encoded(small.getvalue())
    except ValueError:pass
    else:raise AssertionError('Non-reference input size must be rejected.')
    report={'complete':True,'default_version':Predictor().version,'native_fixture_files':len(golden),
        'V1_V3_reference_forward_views':views,'archived_fixture_decisions_checked':decisions,'V2_expert_controls':v2_controls,
        'maximum_reference_logit_difference':maximum,'tolerance_atol':1e-4,'tolerance_rtol':1e-4,
        'bits_and_attribution_decisions_exact':True,'actual_header_route_and_rewrapped_PNG_checked':True,
        'wrong_size_rejected':True,'optimizer_steps':0}
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

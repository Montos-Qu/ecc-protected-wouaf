"""Reassemble checksum-verified local release parts. No network or package installation."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--id',nargs='+',required=True);a=p.parse_args()
    manifest=json.loads((ROOT/'asset_manifest.json').read_text())
    for name in a.id:
        item=manifest['assets'][name];target=(ROOT/item['destination']).resolve()
        assert target.is_relative_to(ROOT.resolve())
        temporary=ROOT/'outputs/asset_cache'/item['original_filename'];temporary.parent.mkdir(parents=True,exist_ok=True)
        with temporary.open('wb') as output:
            for part in item['parts']:
                path=(a.asset_root/part['path']).resolve()
                assert path.is_relative_to(a.asset_root.resolve()) and sha(path)==part['sha256'],str(path)
                with path.open('rb') as source:shutil.copyfileobj(source,output,8*1024*1024)
        assert temporary.stat().st_size==item['bytes'] and sha(temporary)==item['sha256']
        if item['kind']=='zip':
            with zipfile.ZipFile(temporary) as z:
                for member in z.infolist():
                    destination=(target/member.filename).resolve()
                    assert destination.is_relative_to(target),member.filename
                    if member.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(member) as source,destination.open('wb') as output:shutil.copyfileobj(source,output,8*1024*1024)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==item['sha256'],'Existing target differs; refusing overwrite.'
            else:shutil.copy2(temporary,target)
        print('ASSET_READY',name,target)
if __name__=='__main__':main()

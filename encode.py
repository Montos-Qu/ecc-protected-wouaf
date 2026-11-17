"""Encode an integer user ID with the exact published BCH generator matrix."""
import argparse,json
import numpy as np
from ecc_wouaf.core import Decoder,load_config
p=argparse.ArgumentParser();p.add_argument('--code',type=int,choices=[11,16],default=11);p.add_argument('--user-id',type=int,required=True)
a=p.parse_args();assert 0<=a.user_id<2**a.code
book=Decoder(load_config()).books[a.code][2]
message=((np.uint64(a.user_id)>>np.arange(a.code-1,-1,-1,dtype=np.uint64))&1).astype(np.uint8)
word=message@book%2;bits=np.append(word,word.sum()%2).astype(np.uint8)
print(json.dumps({'user_id':a.user_id,'code':f'BCH(31,{a.code})+P','user_bits':message.tolist(),'gt_bits':bits.tolist()}))

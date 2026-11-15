"""Export calibrated INT8 softmax ROM and deterministic golden byte vectors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from softmax_fixed import Config, make_lut, softmax_codes


def main():
    root = Path(__file__).resolve().parent
    folder = root/'runs/int8_21m'
    report = json.loads((folder/'report.json').read_text())
    config = Config(score_scale=report['config']['score_scale'])
    out = root/'example'
    out.mkdir(exist_ok=True)
    (out/'int8_config.json').write_text(json.dumps({'score_scale':config.score_scale},indent=2)+'\n',encoding='utf-8')
    lut = make_lut(config)
    (out/'exp_lut.hex').write_text(''.join(f'{int(v):04x}\n' for v in lut),encoding='ascii')
    rows=[]
    rng=np.random.default_rng(20261114)
    for n in (1,2,3,49,196):
        rows.extend([np.zeros(n,dtype=np.int8),np.full(n,-128,dtype=np.int8),
            np.full(n,127,dtype=np.int8),np.array([127]+[-128]*(n-1),dtype=np.int8),
            np.resize([-128,127],n).astype(np.int8)])
    # Cover every possible ROM address and every legal length.
    rows.extend(np.array([-128,-128+d],dtype=np.int8) for d in range(256))
    for n in range(1,197):
        rows.append(rng.integers(-128,128,n,dtype=np.int16).astype(np.int8))
    # A true even-floor tie distinguishes ties-to-even from the former half-up
    # rule: arrange sum=254*e, so selected output codes are exactly 0.5 -> 0.
    addresses={int(value):i for i,value in enumerate(lut)}
    ties=[]
    for d,value in enumerate(lut):
        e=int(value)
        if e==0:
            continue
        for count in range(1,195):
            other=(254-count)*e-65535
            if other in addresses:
                ties.append((count+2,d,count,addresses[other]))
    if not ties:
        raise ValueError('Need an exact even-floor rounding fixture for this ROM')
    _,d,count,other=min(ties)
    tie=np.array([127]+[127-d]*count+[127-other],dtype=np.int8)
    tie_codes,_=softmax_codes(tie,config)
    np.testing.assert_array_equal(tie_codes[1:-1],np.zeros(count,dtype=np.int8))
    rows.append(tie)
    synthetic=len(rows)
    for record in report['calibration']['images']+report['validation']['images']:
        # Fixed evaluator uses the same artifact name as the reference capture.
        source=json.loads((root/'runs/reference_21m/report.json').read_text())
        item=next(x for x in source['images'] if x['image']==record['image'])
        with np.load(folder/item['artifact'],allow_pickle=False) as data:
            q=data['score_codes'].reshape(-1,data['score_codes'].shape[-1])
            expected=data['probability_codes'].reshape(q.shape)
            for i in rng.choice(len(q),10,replace=False):
                codes,_=softmax_codes(q[i],config)
                np.testing.assert_array_equal(codes,expected[i])
                rows.append(q[i].copy())
    count=sum(map(len,rows))
    with (out/'rows.txt').open('w',encoding='ascii') as stream:
        stream.write(f'{len(rows)} {count}\n')
        for row in rows:
            codes,_=softmax_codes(row,config)
            stream.write(f'{len(row)}\n')
            for q,p in zip(row,codes):
                stream.write(f'{int(q)} {int(p)}\n')
    manifest={'config':config.metadata(),'rows':len(rows),'elements':count,
        'synthetic_rows':synthetic,'real_rows':len(rows)-synthetic,'seed':20261114,
        'exact_even_floor_tie_rows':1,
        'source':'All score differences 0..255, all lengths 1..196, boundaries and 10 actual rows per image; no tuning',
        'rom_sha256':hashlib.sha256((out/'exp_lut.hex').read_bytes()).hexdigest(),
        'vectors_sha256':hashlib.sha256((out/'rows.txt').read_bytes()).hexdigest(),
        'source_report_sha256':hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()

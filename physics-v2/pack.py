import json
from pathlib import Path
def compact(x):
    if isinstance(x,float):return round(x,5)
    if isinstance(x,list):return [compact(v) for v in x]
    if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
    return x
def pack(data):
    return compact(dict(summary=data['summary'],frames=data['frames'][::2]))
if __name__=='__main__':
    out={name:pack(json.loads(Path('results',file+'.json').read_text())) for name,file in [('grasp','grasp'),('zero','zero-force'),('release','release')]}
    Path('tendonhand-physics/recordings.json').write_text(json.dumps(out,separators=(',',':')))

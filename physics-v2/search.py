from concurrent.futures import ProcessPoolExecutor
import random,json
from pathlib import Path
from trial import run
def work(a):
    try:return run(**a)[0]
    except Exception as e:return {'error':str(e),**a}
def score(r):
    if 'error' in r:return -100
    return (100 if r['success'] else 0)+20*r['minHoldDigits']+100*(r['minHoldHeight']-r['initialHeight'])-100*r['maxHoldDrift']
if __name__=='__main__':
    rng=random.Random(101)
    args=[]
    for i in range(1500):
        rad=rng.uniform(.020,.037)
        args.append(dict(center=(rng.uniform(-.017,.017),rng.uniform(.007,.048),rng.uniform(rad+.012,rad+.035)),radius=rad,pull=rng.uniform(3,16),thumb=rng.uniform(2,15),oppose=rng.uniform(-92,-25),friction=rng.uniform(.8,1.6)))
    with ProcessPoolExecutor(max_workers=12) as p:r=list(p.map(work,args,chunksize=3))
    r.sort(key=score,reverse=True)
    Path('results/search.json').write_text(json.dumps(r,indent=2))
    print('success',sum(x.get('success',False) for x in r),'/',len(r))
    print(json.dumps(r[:5],indent=2))

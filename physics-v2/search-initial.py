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
    for i in range(220):
        args.append(dict(center=(rng.uniform(-.012,.012),rng.uniform(.012,.05),rng.uniform(.035,.065)),radius=rng.uniform(.026,.036),pull=rng.uniform(5,15),thumb=rng.uniform(2,11),oppose=rng.uniform(-40,-14),friction=1.15))
    with ProcessPoolExecutor(max_workers=12) as p:r=list(p.map(work,args,chunksize=3))
    r.sort(key=score,reverse=True)
    Path('results/search.json').write_text(json.dumps(r,indent=2))
    print('success',sum(x.get('success',False) for x in r),'/',len(r))
    print(json.dumps(r[:5],indent=2))

"""Blind CPU reference assessment with paired seeds and signed goals."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train_manip import ManipPolicy
from cpu_eval import evaluate
from train import writejson
from scipy.stats import binomtest
def policy(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    p=ManipPolicy();p.load_state_dict(ck['policy']);p.eval();return p,ck
def main(a):
    s,sc=policy('runs/sphere01/best.pt')
    e,ec=policy('runs/ellipsoid01/best.pt')
    seeds=list(range(550000,550000+a.n))
    angles=[(1 if i%2==0 else -1)*[25,40,60][(i//2)%3] for i in range(a.n)]
    out=Path('runs/assessment');out.mkdir(exist_ok=True)
    rows=[]
    for shape in ['sphere','ellipsoid']:
      for diff in [1.,1.5]:
        scores={};trials={}
        for name,p in [('V4_hold',None),('V5_sphere',s),('V5_ellipsoid',e)]:
            score,res,_=evaluate(p,shape,seeds,diff,angles)
            scores[name]=score;trials[name]=res
            print('BLIND_TEST',shape,diff,name,json.dumps(score),flush=True)
        by_goal={}
        for goal in [-60,-40,-25,25,40,60]:
            ids=[i for i,v in enumerate(angles) if v==goal]
            by_goal[str(goal)]={name:float(np.mean([res[i]['success'] for i in ids])) for name,res in trials.items()}
        row=dict(shape=shape,difficulty=diff,goals=angles,score=scores,byGoal=by_goal,trials=trials)
        rows.append(row)
        writejson(out/'blind.json',dict(seeds=seeds,sphereCheckpoint=sc['update'],ellipsoidCheckpoint=ec['update'],results=rows))
        if diff==1.:
            name='V5_sphere' if shape=='sphere' else 'V5_ellipsoid'
            use=s if shape=='sphere' else e
            ids=[i for i,(x,y) in enumerate(zip(trials['V4_hold'],trials[name])) if not x['success'] and y['success']]
            if not ids:ids=[i for i,r in enumerate(trials[name]) if r['success']]
            idx=ids[0] if ids else 0
            for label,p in [('fixed',None),('learned',use)]:
                score,res,frames=evaluate(p,shape,[seeds[idx]],diff,[angles[idx]],True)
                writejson(out/f'{shape}-{label}.json',dict(selection='representative paired blind case; not population estimate',summary=res[0],frames=frames[0]))
    print('BLIND_ASSESSMENT_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=96);main(p.parse_args())

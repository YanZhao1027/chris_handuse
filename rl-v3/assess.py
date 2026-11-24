"""Blind evaluation: checkpoint selection on validation only, unseen seeds at test."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train import Policy,evaluate,writejson
from env import Hand,ACT,OBS,BASE,SCALE,CFG
def load_policy(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    p=Policy();p.load_state_dict(ck['policy']);p.eval();return p,ck
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',default='runs/ppo01');a.add_argument('--n',type=int,default=96);a=a.parse_args()
    path=Path(a.run);policy,ck=load_policy(path/'best.pt')
    seedlist=list(range(20270000,20270000+a.n))
    rows=[]
    for difficulty in [0.3,1.0,1.5]:
        for horizon in [100,150]:
            base,br,_=evaluate(None,seedlist,difficulty,horizon)
            learned,pr,_=evaluate(policy,seedlist,difficulty,horizon)
            row=dict(difficulty=difficulty,horizon=horizon,seconds=horizon*.04,baseline=base,ppo=learned,baselineTrials=br,ppoTrials=pr)
            rows.append(row)
            print('TEST',json.dumps({k:v for k,v in row.items() if not k.endswith('Trials')}),flush=True)
            writejson(path/'test_results.json',dict(checkpointUpdate=ck.get('update'),seeds=seedlist,results=rows))
    # Export representative paired trajectory: first case baseline failed and learned succeeded,
    # else first learned success. The selection is documented, not a statistical evaluation.
    target=next(r for r in rows if r['difficulty']==1. and r['horizon']==150)
    pair=next((b['seed'] for b,p in zip(target['baselineTrials'],target['ppoTrials']) if not b['success'] and p['success']),None)
    if pair is None:pair=next((p['seed'] for p in target['ppoTrials'] if p['success']),seedlist[0])
    for name,pol in [('fixed',None),('learned',policy)]:
        score,res,frames=evaluate(pol,[pair],1.,150,True)
        writejson(path/(name+'_trajectory.json'),dict(selection='representative paired held-out case; chosen for visualization only',summary=res[0],frames=frames[0]))
    print('ASSESSMENT_COMPLETE',json.dumps({'pairedSeed':pair}),flush=True)

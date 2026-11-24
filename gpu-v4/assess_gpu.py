"""Independent CPU reference evaluation of GPU-learned policy (new blind test set)."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gpu_env import V3
from train import Policy,evaluate,writejson
from scipy.stats import binomtest
def model(path):
    ck=torch.load(path,map_location='cpu',weights_only=True);p=Policy();p.load_state_dict(ck['policy']);p.eval();return p,ck
def main(a):
    out=Path(a.run);gpu,ck=model(out/'best.pt');old,_=model(V3/'runs/ppo01/best.pt')
    seeds=list(range(40270000,40270000+a.n))
    report=[]
    for diff in [.3,1.,1.5]:
        base,b,_=evaluate(None,seeds,diff,150)
        cpu,c,_=evaluate(old,seeds,diff,150)
        learned,g,_=evaluate(gpu,seeds,diff,150)
        wins=sum(not x['success'] and y['success'] for x,y in zip(c,g))
        losses=sum(x['success'] and not y['success'] for x,y in zip(c,g))
        r=dict(difficulty=diff,seconds=6.,fixed=base,cpuPPO=cpu,gpuPPO=learned,winsVsCpu=wins,lossesVsCpu=losses,pairedP=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,fixedTrials=b,cpuTrials=c,gpuTrials=g)
        report.append(r);print('BLIND_TEST',json.dumps({k:v for k,v in r.items() if not k.endswith('Trials')}),flush=True)
        writejson(out/'test_results.json',dict(checkpointUpdate=ck['update'],seeds=seeds,results=report))
    case=next((x['seed'] for x,y in zip(report[1]['cpuTrials'],report[1]['gpuTrials']) if not x['success'] and y['success']),None)
    if case is None:case=next((x['seed'] for x in report[1]['gpuTrials'] if x['success']),seeds[0])
    for name,p in [('fixed',None),('cpu',old),('gpu',gpu)]:
        score,trial,frames=evaluate(p,[case],1.,150,True)
        writejson(out/(name+'_trajectory.json'),dict(selection='illustrative paired held-out case',summary=trial[0],frames=frames[0]))
    print('BLIND_TEST_COMPLETE',case,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',default='runs/ppo-gpu01');p.add_argument('--n',type=int,default=96);main(p.parse_args())

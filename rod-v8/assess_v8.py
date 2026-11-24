"""Held-out CPU evaluation; compare hold, searched actions, BC, and PPO."""
import math,json
from pathlib import Path
import numpy as np,torch
from policy_v8 import load_policy
from cpu_eval_rod import evaluate
from train import writejson
torch.set_num_threads(1)
def details(score,rows):
    return dict(score,negative=float(np.mean([r['success'] for r in rows if r['targetDegrees']<0])),positive=float(np.mean([r['success'] for r in rows if r['targetDegrees']>0])))
def main():
    out=Path('runs/assessment');out.mkdir(exist_ok=True)
    teachers=json.loads(Path('runs/teachers.json').read_text())
    prior=torch.tensor([teachers[str(g)][0]['action'] for g in [-15,15]],dtype=torch.float32)
    def fixed(x):return prior[(x[:,-1]>=0).long()]
    bc,_=load_policy('runs/bc.pt');ppo,ck=load_policy('runs/bidir01/best.pt')
    seeds=list(range(812000,812096));results=[]
    for diff,angle in [(.35,15.),(1.,15.),(.35,30.)]:
        goals=[(-1 if i%2==0 else 1)*angle for i in range(len(seeds))]
        scores={};trials={}
        for name,policy in [('hold',None),('searched',fixed),('BC',bc.command),('PPO',ppo.command)]:
            s,r,_=evaluate(policy,seeds,goals,difficulty=diff,scale=2.)
            scores[name]=details(s,r);trials[name]=r
            print('BLIND',diff,angle,name,json.dumps(scores[name]),flush=True)
        row=dict(difficulty=diff,angle=angle,scores=scores,trials=trials)
        results.append(row)
        writejson(out/'blind.json',dict(seeds=seeds,checkpoint=ck.get('update'),results=results))
        if diff==.35 and angle==15.:
            for goal in [-15.,15.]:
                ids=[i for i,g in enumerate(goals) if g==goal]
                i=next((i for i in ids if trials['PPO'][i]['success'] and not trials['hold'][i]['success']),None)
                if i is None:i=next((i for i in ids if trials['PPO'][i]['success']),ids[0])
                for name,policy in [('hold',None),('PPO',ppo.command)]:
                    _,r,f=evaluate(policy,[seeds[i]],[goal],difficulty=diff,scale=2.,record=True)
                    writejson(out/f'{int(goal):+d}-{name}.json',dict(summary=r[0],frames=f[0],selection='illustrative paired held-out trajectory; not population estimate'))
    print('ASSESS_COMPLETE',flush=True)
if __name__=='__main__':main()

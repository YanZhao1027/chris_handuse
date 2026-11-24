"""GPU supervised BC, trajectory-level validation and independent online CPU evaluation."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch,minari
from benchmark import Policy,save,write,evaluate,load
torch.set_num_threads(1)
def prepare(out):
    d=minari.load_dataset('D4RL/pen/expert-v2')
    assert d.total_episodes==4958 and d.total_steps==499206
    ids=np.asarray(d.episode_indices).copy()
    rng=np.random.default_rng(12012);rng.shuffle(ids)
    split=int(.9*len(ids));train=set(ids[:split].tolist())
    tx=[];ty=[];vx=[];vy=[];ret=[];counts=[]
    for e in d.iterate_episodes():
        x=np.asarray(e.observations[:-1],dtype=np.float32);y=np.asarray(e.actions,dtype=np.float32)
        assert x.shape==(len(y),45) and y.shape[1]==24 and np.isfinite(x).all() and np.isfinite(y).all()
        assert np.max(np.abs(y))<=1.000001
        if e.id in train:tx.append(x);ty.append(y)
        else:vx.append(x);vy.append(y)
        o=e.observations[1:]
        # Recover the task's documented goal distance and oriented-axis cosine from observations.
        good=(np.linalg.norm(o[:,39:42],axis=1)<.075)&((o[:,33:36]*o[:,36:39]).sum(1)>.95)
        ret.append(float(np.sum(e.rewards)));counts.append(int(good.sum()))
    arrays=[np.concatenate(v) for v in (tx,ty,vx,vy)]
    assert sum(len(v) for v in arrays[::2])==d.total_steps
    report=dict(dataset='D4RL/pen/expert-v2',totalEpisodes=d.total_episodes,totalSteps=d.total_steps,trainEpisodes=split,validationEpisodes=len(ids)-split,trainSteps=len(arrays[0]),validationSteps=len(arrays[2]),splitSeed=12012,expertMeanRecordedReturn=float(np.mean(ret)),expertSuccess20FromStoredObservations=float(np.mean(np.array(counts)>=20)),expertMeanSuccessSteps=float(np.mean(counts)))
    write(out/'dataset.json',report)
    write(out/'split.json',dict(train=ids[:split].tolist(),validation=ids[split:].tolist()))
    print('DATASET_VERIFIED',json.dumps(report),flush=True)
    return arrays
def main(a):
    torch.manual_seed(12012);torch.cuda.manual_seed_all(12012)
    torch.set_float32_matmul_precision('high')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'history.json').exists()
    tx,ty,vx,vy=prepare(out)
    dev='cuda';x,y,xx,yy=[torch.as_tensor(z,device=dev) for z in (tx,ty,vx,vy)]
    p=Policy().cuda()
    p.mean.copy_(x.mean(0));p.std.copy_(x.std(0).clamp(min=.005))
    opt=torch.optim.AdamW(p.parameters(),lr=3e-4,weight_decay=1e-5)
    seeds=list(range(1202000,1202032));history=[];best=None;t=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        p.train();losses=[]
        for i in torch.randperm(len(x),device=dev).split(2048):
            pred=p(x[i]);loss=(pred-y[i]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1);opt.step()
            losses.append(loss.detach())
        p.eval()
        with torch.inference_mode():
            val=float(torch.cat([(p(xx[i])-yy[i]).square().mean(1) for i in torch.arange(len(xx),device=dev).split(4096)]).mean())
        row=dict(epoch=epoch,trainMSE=float(torch.stack(losses).mean()),validationMSE=val,seconds=time.perf_counter()-t)
        assert np.isfinite(val)
        if epoch==1 or epoch%5==0 or epoch==a.epochs:
            q=Policy();q.load_state_dict({k:v.detach().cpu() for k,v in p.state_dict().items()});q.eval()
            score,_=evaluate(q,seeds,horizon=200);row['onlineValidation']=score
            state=dict(policy=q.state_dict(),epoch=epoch,validation=score,validationMSE=val)
            key=(score['stable20'],score['success20'],score['finalSuccess'],score['meanReturn'],-val)
            if best is None or key>best:
                best=key;save(out/'best.pt',state)
            save(out/'latest.pt',dict(**state,optimizer=opt.state_dict()))
        history.append(row);write(out/'history.json',history);print('EPOCH',json.dumps(row),flush=True)
    q,c=load(out/'best.pt')
    results={}
    for method in ['zero','random','policy']:
        score,rows=evaluate(q,list(range(1203000,1203096)),kind=method)
        results[method]=dict(summary=score,trials=rows);write(out/'blind.json',dict(bestEpoch=c['epoch'],results=results))
        print('BLIND',method,json.dumps(score),flush=True)
    print('BC_COMPLETE',c['epoch'],flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=30);a.add_argument('--out',default='runs/bc01');main(a.parse_args())

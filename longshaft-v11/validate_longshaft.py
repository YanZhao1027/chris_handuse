"""Select the mass for the 220 mm cylinder using calibration only, then blind validation."""
import json,math,concurrent.futures as cf
from pathlib import Path
import numpy as np
import rod_shape as rs
from search_mount import trial
from rod_shape import mount_config
def evaluate(x):
    mass,seed,difficulty=x
    rs.MASS=mass
    r=trial(mount_config(),seed,difficulty)
    r.pop('frames',None)
    return dict(seed=seed,mass=mass,difficulty=difficulty,**r)
def batch(mass,seeds,difficulty):
    with cf.ProcessPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(evaluate,[(mass,s,difficulty) for s in seeds],chunksize=1))
    return dict(n=len(rows),mass=mass,success=sum(r['success'] for r in rows),retained=sum(r['retained'] for r in rows),opposition=float(np.mean([r['opposedFraction'] for r in rows])),maxSelfMM=max(r['maxSelfMM'] for r in rows),maxObjectMM=max(r['maxObjectMM'] for r in rows),trials=rows)
def write(name,obj):
    Path(name).write_text(json.dumps(obj,indent=2))
def main():
    cal=[]
    for mass in [.055,.045,.035]:
        r=batch(mass,range(1101000,1101024),.35)
        cal.append(r)
        print('CALIBRATION',json.dumps({k:v for k,v in r.items() if k!='trials'}),flush=True)
        if r['success']>=23:break
    write('runs/mass-calibration.json',cal)
    best=max(cal,key=lambda r:(r['success'],r['retained'],r['mass']))
    if best['success']<22:raise RuntimeError('Grasp calibration insufficient')
    m=best['mass']
    report=dict(lengthMM=220,diameterMM=26,massG=m*1000,densityKgM3=m/(math.pi*.013**2*.220),inertiaKgM2=rs.inertia(m,.013,.110).tolist(),calibration={k:v for k,v in best.items() if k!='trials'},blind=[])
    for diff,seed in [(.35,1102000),(1.,1103000)]:
        r=batch(m,range(seed,seed+64),diff)
        report['blind'].append(dict(difficulty=diff,**r))
        print('BLIND',diff,json.dumps({k:v for k,v in r.items() if k!='trials'}),flush=True)
        write('runs/validation.json',report)
    if report['blind'][0]['success']<58:raise RuntimeError('Long shaft holdout gate failed')
    print('SELECTED',m,'kg',report['densityKgM3'],'kg/m^3',flush=True)
if __name__=='__main__':main()

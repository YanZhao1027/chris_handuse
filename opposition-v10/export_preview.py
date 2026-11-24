"""Render exact MuJoCo geometry transforms, including opposed thumb mount."""
import json
from pathlib import Path
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
from rod_shape import RodHand,mount_config,audit_pairs
from search_mount import trial
seed=1003000;c=mount_config()
r=trial(c,seed,.35,record=True)
h=RodHand(c);h.reset(seed,.35);m=h.m;d=h.d
names=['palm','drive']+[f'{n}_segment{j}' for n in ['index','middle','ring','little','thumb'] for j in range(3)]+['ball']
ids=[m.geom(n).id for n in names]
types={int(mj.mjtGeom.mjGEOM_BOX):'box',int(mj.mjtGeom.mjGEOM_CAPSULE):'capsule',int(mj.mjtGeom.mjGEOM_CYLINDER):'cylinder'}
geoms=[dict(name=n,type=types[int(m.geom_type[i])],size=(1000*m.geom_size[i]).tolist()) for n,i in zip(names,ids)]
pairs=audit_pairs(m);worst=0.;objectworst=0.;frames=r.pop('frames')
for f in frames:
    d.qpos[h.qj[:,0]]=np.radians(f['yaw']);d.qpos[h.qj[:,1:]]=np.radians(f['angles'])
    d.qpos[h.obj_q:h.obj_q+3]=f['object'];d.qpos[h.obj_q+3:h.obj_q+7]=f['objectQuat'];d.mocap_pos[h.wrist]=f['wrist']
    d.eq_active[h.eq]=f['fixture']
    mj.mj_forward(m,d)
    f['poses']=[dict(p=(1000*d.geom_xpos[i]).tolist(),q=Rotation.from_matrix(d.geom_xmat[i].reshape(3,3)).as_quat().tolist()) for i in ids]
    worst=max(worst,-min(mj.mj_geomDistance(m,d,a,b,.2,None) for a,b in pairs))
    objectworst=max(objectworst,-min(mj.mj_geomDistance(m,d,i,h.ball,.2,None) for i in ids[:-1]))
r['geometryAudit']=dict(maxSelfPenetrationMM=1000*worst,maxRodPenetrationMM=1000*objectworst)
assert r['success'] and worst<.001 and objectworst<.002,r
out=dict(seed=seed,summary=r,geoms=geoms,frames=frames,mount=c,params=h.params)
Path('handuse-opposition-preview/recordings.json').write_text(json.dumps(out,separators=(',',':')))
Path('runs/preview-audit.json').write_text(json.dumps(dict(seed=seed,**r),indent=2))
print('EXACT_GEOMETRY_PREVIEW',json.dumps(dict(seed=seed,**r)),flush=True)

"""Batched MuJoCo Warp residual-RL environment, GPU tensors throughout rollout."""
import sys,time
from pathlib import Path
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
V3=Path(__file__).resolve().parent.parent/'rl-v3'
sys.path.insert(0,str(V3))
from env import Hand,CFG,ACT,OBS,BASE,SCALE,DT,SUB,START_YAW,domain

@wp.kernel(enable_backward=False)
def control(clock:wp.array(dtype=wp.float32),u:wp.array2d(dtype=wp.float32),
            ctrl:wp.array2d(dtype=wp.float32),mocap:wp.array2d(dtype=wp.vec3),
            yaws:wp.array(dtype=wp.int32),pulls:wp.array(dtype=wp.int32),wrist:int):
    w=wp.tid()
    t=clock[w]
    a=wp.clamp(t/0.70,0.0,1.0);a=a*a*(3.0-2.0*a)
    l=wp.clamp((t-1.10)/0.85,0.0,1.0);l=l*l*(3.0-2.0*l)
    for i in range(5):
        s=float(0.0);g=float(0.0)
        if i==0:s=8.0;g=2.0
        elif i==1:s=3.0;g=1.0
        elif i==2:s=-3.0;g=-1.0
        elif i==3:s=-10.0;g=-3.0
        else:s=62.0;g=u[w,5]
        ctrl[w,yaws[i]]=0.017453292519943295*((1.0-a)*s+a*g)
        ctrl[w,pulls[i]]=a*u[w,i]
    mocap[w,wrist]=wp.vec3(0.0,0.0,0.25+0.065*l)

@wp.kernel(enable_backward=False)
def reduce_contacts(nacon:wp.array(dtype=wp.int32),geom:wp.array(dtype=wp.vec2i),
                    world:wp.array(dtype=wp.int32),f:wp.array(dtype=wp.spatial_vector),
                    mapping:wp.array(dtype=wp.int32),ball:int,floor:int,
                    per_digit:wp.array2d(dtype=wp.float32),total:wp.array(dtype=wp.float32)):
    i=wp.tid()
    if i>=nacon[0]:return
    g=geom[i];other=int(-1)
    if g[0]==ball:other=g[1]
    elif g[1]==ball:other=g[0]
    if other<0 or other==floor:return
    w=world[i];value=wp.max(0.0,f[i][0])
    wp.atomic_add(total,w,value)
    digit=mapping[other]
    if digit>=0:wp.atomic_add(per_digit,w,digit,value)

def tensor(x,dtype=torch.float32):
    return torch.as_tensor(x,device='cuda:0',dtype=dtype)

class GPUHand:
    def __init__(self,n,seed=20261124,difficulty=1.,variants=256,horizon=100,exact_seeds=None,iterations=80):
        self.n=n;self.horizon=horizon;self.difficulty=difficulty
        self.generator=torch.Generator(device='cuda:0');self.generator.manual_seed(seed)
        wp.set_device('cuda:0')
        self.stream=wp.Stream(device='cuda:0');wp.set_stream(self.stream)
        torch.cuda.set_stream(wp.stream_to_torch(self.stream))
        h=Hand();h.reset(seed,0)
        h.m.opt.iterations=iterations
        self.cpu=h;self.m=mw.put_model(h.m)
        # A pool of independently compiled physical parameter sets. Warp indexes leading
        # model dimensions modulo their size. Every world gets one pool member.
        self.variants=n if exact_seeds is not None else min(n,variants)
        seeds=exact_seeds if exact_seeds is not None else [seed+37*j for j in range(self.variants)]
        fields=['body_mass','body_inertia','body_invweight0','body_subtreemass','dof_invweight0','geom_size','geom_aabb','geom_rbound','geom_friction','actuator_acc0','tendon_invweight0']
        params=[];data={f:[] for f in fields}
        for s in seeds:
            h.reset(s,difficulty);params.append(h.params)
            for f in fields:data[f].append(np.array(getattr(h.m,f),copy=True))
        for f in fields:
            a=getattr(self.m,f)
            vals=np.stack(data[f]).astype(np.float32)
            if f=='geom_aabb':vals=vals.reshape(self.variants,h.m.ngeom,2,3)
            setattr(self.m,f,wp.array(vals,dtype=a.dtype,device='cuda:0'))
        # Reset the CPU template to the nominal state; per-world settings live on GPU.
        h.reset(seed,0)
        self.d=mw.put_data(h.m,h.d,nworld=n,nconmax=32,njmax=128)
        d=self.d
        self.q=wp.to_torch(d.qpos);self.v=wp.to_torch(d.qvel)
        self.ctrl=wp.to_torch(d.ctrl);self.eq=wp.to_torch(d.eq_active)
        self.mocap=wp.to_torch(d.mocap_pos);self.ten=wp.to_torch(d.ten_length)
        self.qids=tensor(h.qj.ravel(),torch.long);self.vids=tensor(h.vj.ravel(),torch.long)
        self.qyaw=tensor(h.qj[:,0],torch.long);self.yawids=tensor(h.yaws,torch.long)
        self.start=tensor(START_YAW);self.base=tensor(BASE);self.scale=tensor(SCALE)
        self.sign=tensor([-1.,1.,-1.])
        self.wrist=h.wrist;self.fix=h.fix;self.obj_q=h.obj_q;self.obj_v=h.obj_v
        self.eqid=h.eq;self.ball=h.ball;self.floor=h.floor
        self.yaws=wp.array(h.yaws.astype(np.int32),dtype=wp.int32)
        self.pulls=wp.array(h.pulls.astype(np.int32),dtype=wp.int32)
        mp=np.full(h.m.ngeom,-1,dtype=np.int32)
        for k,v in h.geom_digit.items():mp[k]=v
        self.mapping=wp.array(mp,dtype=wp.int32)
        pool=torch.arange(n,device='cuda:0')%self.variants
        self.centers0=tensor([p['center'] for p in params])[pool]
        self.radius=tensor([p['radius'] for p in params])[pool]
        self.center=self.centers0.clone()
        self.mass=tensor([p['mass'] for p in params])[pool]
        self.friction=tensor([p['friction'] for p in params])[pool]
        self.exact=exact_seeds is not None
        self.last=torch.zeros((n,ACT),device='cuda')
        self.u=self.base.repeat(n,1).contiguous();self.u_wp=wp.from_torch(self.u,dtype=wp.float32)
        self.k=torch.zeros(n,device='cuda',dtype=torch.int32)
        self.release_center=self.center.clone()
        self.minheight=torch.ones(n,device='cuda');self.maxdrift=torch.zeros(n,device='cuda')
        self.mindigits=torch.full((n,),5,device='cuda',dtype=torch.int32)
        self.returns=torch.zeros(n,device='cuda')
        self.forces=wp.zeros((n,5),dtype=wp.float32)
        self.total=wp.zeros(n,dtype=wp.float32)
        self.force_t=wp.to_torch(self.forces);self.total_t=wp.to_torch(self.total)
        self.contact_ids=wp.array(np.arange(d.naconmax,dtype=np.int32),dtype=wp.int32)
        self.contact_force=wp.zeros(d.naconmax,dtype=wp.spatial_vector)
        self.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
        # Warm up solver/control/reduction before capture. All launches share the torch stream.
        self._substep();self._contacts();wp.synchronize()
        # Capture one integration step (contains Warp's conditional Newton graph).
        with wp.ScopedCapture() as cap:self._substep()
        self.graph=cap.graph
        self.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
        wp.synchronize()
        print('GPU_ENV_READY',n,'worlds','variants',self.variants,'iterations',iterations,flush=True)
    def _substep(self):
        wp.launch(control,dim=self.n,inputs=[self.d.time,self.u_wp,self.d.ctrl,self.d.mocap_pos,self.yaws,self.pulls,self.wrist])
        mw.step(self.m,self.d)
    def _contacts(self):
        self.forces.zero_();self.total.zero_()
        mw.contact_force(self.m,self.d,self.contact_ids,False,self.contact_force)
        wp.launch(reduce_contacts,dim=self.d.naconmax,inputs=[self.d.nacon,self.d.contact.geom,self.d.contact.worldid,self.contact_force,self.mapping,self.ball,self.floor,self.forces,self.total])
    def relative(self):
        return (self.q[:,self.obj_q:self.obj_q+3]-self.mocap[:,self.wrist])*self.sign
    def reset(self,mask,initial=False):
        # Stable data buffers (never replace captured pointers).
        if not bool(mask.any().item()):return
        mw.reset_data(self.m,self.d,wp.from_torch(mask.contiguous(),dtype=wp.bool))
        ids=mask.nonzero(as_tuple=True)[0]
        if self.exact or initial:new=self.centers0[ids]
        else:
            # Resample object position each episode; mass/radius/friction fixed per world
            # across this run, randomized across pool members.
            rand=torch.rand((len(ids),3),generator=self.generator,device='cuda')*2-1
            new=tensor(CFG['center'])+rand*tensor([.0035,.0035,.003])*self.difficulty
        self.center[ids]=new
        self.release_center[ids]=new
        pos=new*self.sign;pos[:,2]+=.25
        self.q[ids,self.obj_q:self.obj_q+3]=pos
        self.q[ids,self.obj_q+3:self.obj_q+7]=tensor([1.,0.,0.,0.])
        self.q[ids[:,None],self.qyaw[None,:]]=self.start
        self.ctrl[ids[:,None],self.yawids[None,:]]=self.start
        self.mocap[ids,self.wrist]=tensor([0.,0.,.25])
        self.mocap[ids,self.fix]=pos
        self.eq[ids,self.eqid]=True
        self.k[ids]=0;self.last[ids]=0;self.u[ids]=self.base
        self.minheight[ids]=1;self.maxdrift[ids]=0;self.mindigits[ids]=5
        self.returns[ids]=0
        self.force_t[ids]=0;self.total_t[ids]=0;self.ten[ids]=0
    def obs(self):
        rel=self.relative()
        vel=self.v[:,self.obj_v:self.obj_v+3]*self.sign
        f=self.force_t
        phase=(self.k.float()*DT/2).clamp(max=1.5)
        o=torch.cat((self.q[:,self.qids]/1.5,(self.v[:,self.vids]/15).clamp(-3,3),rel/.065,(vel/.5).clamp(-3,3),self.center/.065,(f/4).clamp(0,5),self.ten/.025,self.last,phase[:,None],self.eq[:,self.eqid,None].float(),self.radius[:,None]/.025),1)
        assert o.shape==(self.n,OBS),o.shape
        return o.clamp(-10,10)
    @torch.no_grad()
    def step(self,action,autoreset=True):
        raw=action.clamp(-1,1)
        self.last.mul_(.65).add_(raw,alpha=.35)
        self.u.copy_(self.base+self.scale*self.last)
        self.u[:,:5].clamp_(0,25);self.u[:,5].clamp_(-95,65)
        release=self.k==25
        self.release_center.copy_(torch.where(release[:,None],self.relative(),self.release_center))
        self.eq[:,self.eqid].logical_and_(~release)
        for _ in range(SUB):wp.capture_launch(self.graph)
        self.k.add_(1)
        self._contacts()
        z=self.q[:,self.obj_q+2]
        rel=self.relative();dist=(rel-self.release_center).norm(dim=1)
        n=(self.force_t>.005).sum(1).int()
        thumb=(self.force_t[:,-1]>.005).float()
        hold=self.k>=50
        self.minheight.copy_(torch.where(hold,torch.minimum(self.minheight,z),self.minheight))
        self.maxdrift.copy_(torch.where(hold,torch.maximum(self.maxdrift,dist),self.maxdrift))
        self.mindigits.copy_(torch.where(hold,torch.minimum(self.mindigits,n),self.mindigits))
        excess=(self.total_t-12).clamp(min=0)
        initial=.25-self.center[:,2]
        pre=.025*n+.025*thumb-.003*excess
        post=.45*((z-(initial-.045))/.045).clamp(0,1)+.55*torch.exp(-dist/.020)+.10*n.clamp(max=4)/4+.07*thumb-.008*excess
        reward=torch.where(self.k<=25,pre,post)-.012*(raw**2).mean(1)
        bad=~torch.isfinite(self.q).all(1)
        dropped=(self.k>30)&((z<.09)|(dist>.15))
        done=bad|dropped|(self.k>=self.horizon)
        success=~bad&~dropped&(self.k>=self.horizon)&(self.mindigits>=2)&(self.maxdrift<.025)&(self.minheight>initial+.035)&~self.eq[:,self.eqid]
        reward=reward+torch.where(done,torch.where(success,8.,-5.),0.)
        reward=torch.nan_to_num(reward,nan=-5.,posinf=-5.,neginf=-5.)
        self.returns.add_(reward)
        info={'done':done,'success':success,'return':self.returns.clone(),'bad':bad,'finalHeight':z.clone(),'minheight':self.minheight.clone(),'maxdrift':self.maxdrift.clone(),'mindigits':self.mindigits.clone()}
        if autoreset:self.reset(done)
        return self.obs(),reward,done,info
    def capacities(self):
        return {'contacts':int(wp.to_torch(self.d.nacon).item()),'contactCapacity':self.d.naconmax,'maxConstraints':int(wp.to_torch(self.d.nefc).max().item()),'constraintCapacity':self.d.njmax,'finite':bool(torch.isfinite(self.q).all().item())}

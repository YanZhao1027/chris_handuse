"""E3M5 batched free-body manipulation, with per-world mass/friction variants."""
import json,math
from pathlib import Path
import numpy as np,torch,mujoco as mj,mujoco_warp as mw,warp as wp
from model import make_model,ids,digitmap
from common import *
from contact_gpu import reduce_contacts,penetration,reduce_opposition
def tensor(x,dtype=torch.float32):return torch.as_tensor(x,device='cuda:0',dtype=dtype)
class Env:
    def __init__(self,n,seed=13013,variants=256):
        self.n=n;self.gen=torch.Generator(device='cuda').manual_seed(seed)
        wp.set_device('cuda:0');self.stream=wp.Stream(device='cuda:0');wp.set_stream(self.stream);torch.cuda.set_stream(wp.stream_to_torch(self.stream))
        m=make_model();d=mj.MjData(m);j=ids(m);self.j=j;self.cpu=m
        mj.mj_forward(m,d)
        self.m=mw.put_model(m)
        self.variants=min(n,variants)
        rng=np.random.default_rng(seed);mass0=float(m.body_mass[j['rod']]);inertia0=m.body_inertia[j['rod']].copy()
        fs=['body_mass','body_inertia','body_invweight0','body_subtreemass','dof_invweight0','geom_friction','actuator_acc0','tendon_invweight0']
        data={f:[] for f in fs};pars=[]
        for i in range(self.variants):
            mass=mass0*rng.uniform(.80,1.20);fric=rng.uniform(.85,1.30)
            m.body_mass[j['rod']]=mass;m.body_inertia[j['rod']]=inertia0*mass/mass0;m.geom_friction[j['shaft'],0]=fric
            mj.mj_setConst(m,d)
            for f in fs:data[f].append(np.array(getattr(m,f),copy=True))
            pars.append([mass,fric])
        for f in fs:
            a=getattr(self.m,f);setattr(self.m,f,wp.array(np.stack(data[f]).astype(np.float32),dtype=a.dtype,device='cuda:0'))
        m.body_mass[j['rod']]=mass0;m.body_inertia[j['rod']]=inertia0;m.geom_friction[j['shaft'],0]=1.1;mj.mj_setConst(m,d)
        self.d=mw.put_data(m,d,nworld=n,nconmax=128,njmax=512)
        self.q=wp.to_torch(self.d.qpos);self.v=wp.to_torch(self.d.qvel);self.ctrl=wp.to_torch(self.d.ctrl)
        self.xmat=wp.to_torch(self.d.xmat);self.xpos=wp.to_torch(self.d.xpos)
        self.nefc=wp.to_torch(self.d.nefc)
        self.lo=tensor(m.actuator_ctrlrange[:,0]);self.hi=tensor(m.actuator_ctrlrange[:,1])
        p=tensor(pars)[torch.arange(n,device='cuda')%self.variants];self.mass=p[:,0];self.friction=p[:,1]
        bank=json.loads(Path('runs/reset-bank.json').read_text())
        self.bankq=tensor([x['qpos'] for x in bank]);self.bankv=tensor([x['qvel'] for x in bank]);self.bankc=tensor([x['ctrl'] for x in bank])
        self.base=torch.zeros((n,20),device='cuda');self.last=torch.zeros((n,ACT),device='cuda')
        self.k=torch.zeros(n,device='cuda',dtype=torch.int32);self.streak=torch.zeros_like(self.k)
        self.goal=torch.zeros((n,3),device='cuda');self.startpos=torch.zeros_like(self.goal)
        self.angle=torch.zeros(n,device='cuda');self.prev_err=torch.zeros_like(self.angle);self.maxdrift=torch.zeros_like(self.angle)
        self.oppsteps=torch.zeros_like(self.angle);self.maxself=torch.zeros_like(self.angle);self.maxobj=torch.zeros_like(self.angle)
        self.goal_min=15.;self.goal_max=20.;self.multi_fraction=0.
        self.plane=torch.zeros(n,device='cuda',dtype=torch.long)
        self.released=torch.zeros((n,5),device='cuda',dtype=torch.bool)
        self.ever_loaded=torch.zeros_like(self.released)
        self.off=torch.zeros((n,5),device='cuda',dtype=torch.int32)
        self.on=torch.zeros_like(self.off)
        self.recontacts=torch.zeros_like(self.off)
        self.mapping=wp.array(digitmap(m),dtype=wp.int32)
        self.force=wp.zeros((n,5),dtype=wp.float32);self.total=wp.zeros(n,dtype=wp.float32)
        self.f=wp.to_torch(self.force);self.ft=wp.to_torch(self.total)
        self.sd=wp.zeros(n,dtype=wp.float32);self.od=wp.zeros(n,dtype=wp.float32);self.selfpen=wp.to_torch(self.sd);self.objpen=wp.to_torch(self.od)
        self.normal=wp.zeros((n,2,3),dtype=wp.float32);self.loads=wp.zeros((n,2),dtype=wp.float32)
        self.nt=wp.to_torch(self.normal);self.lt=wp.to_torch(self.loads)
        self.cos=torch.ones(n,device='cuda');self.opposed=torch.zeros(n,device='cuda',dtype=torch.bool)
        self.contact_ids=wp.array(np.arange(self.d.naconmax,dtype=np.int32),dtype=wp.int32)
        self.contact_force=wp.zeros(self.d.naconmax,dtype=wp.spatial_vector)
        self.reset(torch.ones(n,device='cuda',dtype=torch.bool))
        # Compile, then capture full integration and per-substep penetration audit.
        self.substep();self.contacts();wp.synchronize()
        with wp.ScopedCapture() as cap:self.substep()
        self.graph=cap.graph
        self.reset(torch.ones(n,device='cuda',dtype=torch.bool))
        wp.synchronize()
        print('ENV_READY',n,'worlds',self.variants,'variants',len(bank),'training reset poses','contact capacity',self.d.naconmax,'freeGiB',torch.cuda.mem_get_info()[0]/2**30,flush=True)
    def rel(self):
        j=self.j;return relative(self.xmat[:,j['palm']],self.xpos[:,j['palm']],self.xpos[:,j['rod']],self.xmat[:,j['rod'],:,2])
    def substep(self):
        mw.step(self.m,self.d)
        wp.launch(penetration,dim=self.d.naconmax,inputs=[self.d.nacon,self.d.contact.geom,self.d.contact.worldid,self.d.contact.dist,self.j['shaft'],self.j['floor'],self.sd,self.od])
    def contacts(self):
        self.force.zero_();self.total.zero_();self.normal.zero_();self.loads.zero_()
        j=self.j;d=self.d
        mw.contact_force(self.m,d,self.contact_ids,False,self.contact_force)
        wp.launch(reduce_contacts,dim=d.naconmax,inputs=[d.nacon,d.contact.geom,d.contact.worldid,self.contact_force,self.mapping,j['shaft'],j['floor'],self.force,self.total])
        wp.launch(reduce_opposition,dim=d.naconmax,inputs=[d.nacon,d.contact.geom,d.contact.worldid,d.contact.frame,self.contact_force,d.xmat,self.mapping,j['shaft'],j['rod'],self.normal,self.loads])
        lens=self.nt.norm(dim=2);cos=(self.nt[:,0]*self.nt[:,1]).sum(1)/(lens[:,0]*lens[:,1]).clamp(min=1e-9)
        self.cos.copy_(torch.where(lens.min(1).values>.02,cos.clamp(-1.,1.),1.))
        self.opposed.copy_((self.cos<-.35)&(self.lt.min(1).values>.1)&((self.f[:,:4]>.005).sum(1)>=2))
    @torch.no_grad()
    def reset(self,mask):
        ii=mask.nonzero(as_tuple=True)[0]
        if not len(ii):return
        mw.reset_data(self.m,self.d,wp.from_torch(mask.contiguous(),dtype=wp.bool))
        b=torch.randint(len(self.bankq),(len(ii),),generator=self.gen,device='cuda')
        self.q[ii]=self.bankq[b];self.v[ii]=self.bankv[b];self.base[ii]=self.bankc[b];self.ctrl[ii]=self.base[ii]
        self.last[ii]=0;self.k[ii]=0;self.streak[ii]=0;self.maxdrift[ii]=0;self.oppsteps[ii]=0;self.maxself[ii]=0;self.maxobj[ii]=0
        # Keep a quarter of episodes at the established +/-15-degree target.
        u=torch.rand(len(ii),device='cuda',generator=self.gen)
        anchor=u<.25
        frontier=self.goal_min+torch.rand(len(ii),device='cuda',generator=self.gen)*(self.goal_max-self.goal_min)
        self.plane[ii]=((torch.rand(len(ii),device='cuda',generator=self.gen)<self.multi_fraction/.75)&~anchor).long()
        # Secondary-plane training starts small, independently of planar frontier.
        frontier=torch.where(self.plane[ii].bool(),8.+torch.rand(len(ii),device='cuda',generator=self.gen)*min(22.,self.goal_max-8.),frontier)
        magnitude=torch.where(anchor,15.,frontier)
        self.angle[ii]=magnitude*math.pi/180*torch.where(torch.rand(len(ii),device='cuda',generator=self.gen)>.5,1.,-1.)
        self.off[ii]=0;self.on[ii]=0;self.recontacts[ii]=0;self.released[ii]=False;self.ever_loaded[ii]=False
        # Recompute body transforms and contact forces from the reset physical state.
        mw.forward(self.m,self.d);self.contacts()
        pos,axis=self.rel();self.startpos[ii]=pos[ii];self.goal[ii]=target(axis[ii],self.angle[ii],self.plane[ii]);self.ever_loaded[ii]=self.f[ii]>.10;self.prev_err[ii]=self.angle[ii].abs()
        self.selfpen[ii]=0;self.objpen[ii]=0
    def obs(self):
        pos,axis=self.rel()
        return obs(self.q,self.v,pos,axis,self.goal,self.last,self.f,self.cos,self.k.float(),self.mass,self.friction)
    @torch.no_grad()
    def step(self,a,autoreset=True):
        self.last.mul_(.55).add_(a.clamp(-1,1),alpha=.45)
        self.ctrl.copy_(controls(self.base,self.last,self.lo,self.hi))
        self.sd.zero_();self.od.zero_()
        for _ in range(SUB):wp.capture_launch(self.graph)
        self.contacts();self.k.add_(1)
        # Recontact is a measured contact transition, not a proxy from joint motion.
        low=self.f<.05;high=self.f>.10
        self.off.copy_(torch.where(low,self.off+1,0))
        self.ever_loaded.logical_or_(high)
        self.released.logical_or_((self.off>=5)&self.ever_loaded)
        self.on.copy_(torch.where(high,self.on+1,0))
        back=self.released&(self.on>=5)
        self.recontacts.add_(back.int());self.released.logical_and_(~back)
        self.maxself.copy_(torch.maximum(self.maxself,self.selfpen));self.maxobj.copy_(torch.maximum(self.maxobj,self.objpen))
        pos,axis=self.rel();err=error(axis,self.goal);dist=(pos-self.startpos).norm(dim=1);omega=self.v[:,27:30].norm(dim=1)
        self.maxdrift.copy_(torch.maximum(self.maxdrift,dist));self.oppsteps.add_(self.opposed.float())
        bad=~torch.isfinite(self.q).all(1)|~torch.isfinite(self.v).all(1)
        dropped=(self.q[:,26]<.10)|(dist>.085)
        near=(err<math.radians(5))&(dist<.035)&(omega<1.4)&self.opposed
        self.streak.copy_(torch.where(near,self.streak+1,0))
        safe=(self.maxself<.0015)&(self.maxobj<.003)
        end=self.k>=HORIZON;done=bad|dropped|end
        retained=end&~bad&~dropped&(self.maxdrift<.04)
        success=retained&(self.streak>=20)&(self.oppsteps/self.k.clamp(min=1)>=.75)&safe
        reward=.6*torch.exp(-dist/.025)+.40*self.opposed.float()+.10*(-self.cos).clamp(0,1)+3.5*torch.exp(-err/.22)+5*(self.prev_err-err).clamp(-.2,.2)+.3*near.float()-.025*omega.clamp(max=20)-.025*(self.last**2).mean(1)-.4*(self.selfpen/.001).clamp(max=8)-.15*(self.objpen/.002).clamp(max=8)
        reward+=torch.where(done,torch.where(success,12.,torch.where(retained,1.,-12.)),0.)
        self.prev_err.copy_(err)
        info=dict(success=success,retained=retained,done=done,bad=bad,error=err,angle=self.angle.clone(),opposition=self.oppsteps/self.k.clamp(min=1),selfMM=self.maxself.clone()*1000,objMM=self.maxobj.clone()*1000,driftMM=self.maxdrift.clone()*1000,opposed=self.opposed.clone(),dropped=dropped,plane=self.plane.clone(),recontacts=self.recontacts.sum(1).clone(),recontactDigits=(self.recontacts>0).sum(1).clone())
        reward=torch.nan_to_num(reward,nan=-12.,posinf=-12.,neginf=-12.)
        if autoreset:self.reset(done)
        return self.obs(),reward,done,info
    def capacity(self):
        return dict(contacts=int(wp.to_torch(self.d.nacon).item()),contactCapacity=self.d.naconmax,constraints=int(self.nefc.max().item()),constraintCapacity=self.d.njmax,finite=bool(torch.isfinite(self.q).all()),freeGiB=torch.cuda.mem_get_info()[0]/2**30)

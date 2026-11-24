"""Goal-conditioned rod reorientation with a frozen V4 grasp prior."""
import math
from pathlib import Path
import torch,warp as wp
from gpu_rod_base import GPUHand,tensor,OBS,ACT,BASE,SCALE,DT,SUB
from train import Policy
MAN_ACT=10
MAN_OBS=OBS+4+3+10+3+3+1+1
V4=Path(__file__).resolve().parent.parent/'gpu-v4'
PRETRAINED=V4/'runs/ppo-gpu01/best.pt'
START_MAN=50
HORIZON=150
def quat_mul(a,b):
    aw,av=a[:,:1],a[:,1:];bw,bv=b[:,:1],b[:,1:]
    return torch.cat((aw*bw-(av*bv).sum(1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=1)),1)
def qerror(target,q):
    q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-8)
    e=quat_mul(target,torch.cat((q[:,:1],-q[:,1:]),1))
    e=e*torch.where(e[:,:1]<0,-1.,1.)
    angle=2*torch.atan2(e[:,1:].norm(dim=1),e[:,0].abs().clamp(min=1e-8))
    return e,angle
class RodManipHand(GPUHand):
    def __init__(self,n,seed=2026112406,difficulty=1.,variants=1024,horizon=HORIZON,exact_seeds=None,goal_min=8.,goal_max=25.,iterations=80):
        self.man_ready=False
        super().__init__(n,seed,difficulty,variants,horizon,exact_seeds,iterations,'rod')
        self.goal_min=goal_min;self.goal_max=goal_max
        
        self.mlast=torch.zeros((n,MAN_ACT),device='cuda')
        self.mscale=tensor([5.,5.,5.,5.,4.,20.,20.,20.,20.,30.])
        self.target=torch.zeros((n,4),device='cuda');self.target[:,0]=1
        self.target_angle=torch.zeros(n,device='cuda');self.prev_error=torch.zeros(n,device='cuda')
        self.started=torch.zeros(n,dtype=torch.bool,device='cuda');self.grasped=torch.zeros_like(self.started)
        self.good_streak=torch.zeros(n,dtype=torch.int32,device='cuda')
        self.best_error=torch.full((n,),math.pi,device='cuda')
        self.lifted_min=torch.ones(n,device='cuda');self.hold_contacts=torch.full((n,),5,device='cuda',dtype=torch.int32)
        self.max_self=torch.zeros(n,device="cuda");self.max_obj=torch.zeros(n,device="cuda")
        self.opp_count=torch.zeros(n,device="cuda");self.active_count=torch.zeros(n,device="cuda")
        self.man_ready=True;self.fixed_goals=None
        self.reset(torch.ones(n,dtype=torch.bool,device='cuda'),initial=True)
    def reset(self,mask,initial=False):
        super().reset(mask,initial)
        if not self.man_ready:return
        ids=mask.nonzero(as_tuple=True)[0]
        if len(ids)==0:return
        self.mlast[ids]=0;self.started[ids]=False;self.grasped[ids]=False;self.max_self[ids]=0;self.max_obj[ids]=0;self.opp_count[ids]=0;self.active_count[ids]=0
        self.good_streak[ids]=0;self.best_error[ids]=math.pi;self.lifted_min[ids]=1;self.hold_contacts[ids]=5
        self.target[ids]=tensor([1.,0.,0.,0.])
        if self.fixed_goals is not None:self.target_angle[ids]=self.fixed_goals[ids]
        else:
            a=torch.rand(len(ids),device='cuda',generator=self.generator)
            sign=torch.where(torch.rand(len(ids),device='cuda',generator=self.generator)>.5,1.,-1.)
            self.target_angle[ids]=sign*(self.goal_min+(self.goal_max-self.goal_min)*a)*math.pi/180
        self.prev_error[ids]=self.target_angle[ids].abs()
    def quat(self):return self.q[:,self.obj_q+3:self.obj_q+7]
    def obs(self):
        if not self.man_ready:return super().obs()
        e,_=qerror(self.target,self.quat())
        pending=torch.zeros_like(e);pending[:,0]=torch.cos(self.target_angle/2);pending[:,3]=torch.sin(self.target_angle/2)
        e=torch.where(self.started[:,None],e,pending)
        omega=self.v[:,self.obj_v+3:self.obj_v+6]
        extra=torch.cat((e,(omega/8).clamp(-3,3),self.mlast,self.axes/.05,self.opp_cos[:,None],(self.load_t/4).clamp(0,5),((self.k.float()-START_MAN)*DT/4).clamp(0,1)[:,None],self.target_angle[:,None]/math.pi),1)
        o=torch.cat((super().obs(),extra),1);assert o.shape==(self.n,MAN_OBS),o.shape
        return o.clamp(-10,10)
    @torch.no_grad()
    def step(self,action,autoreset=True,active_mask=None):
        raw=action.clamp(-1,1)
        if active_mask is None:active_mask=torch.ones(self.n,dtype=torch.bool,device='cuda')
        initiate=(self.k==START_MAN)&active_mask
        q=self.quat().clone();dq=torch.zeros_like(q);dq[:,0]=torch.cos(self.target_angle/2);dq[:,3]=torch.sin(self.target_angle/2)
        self.target.copy_(torch.where(initiate[:,None],quat_mul(dq,q),self.target));self.started.logical_or_(initiate)
        good_grasp=(self.q[:,self.obj_q+2]>(.25-self.center[:,2]+.035))&((self.force_t>.005).sum(1)>=2)&~self.eq[:,self.eqid]&self.opposed
        self.grasped.logical_or_(initiate&good_grasp)
        self.prev_error.copy_(torch.where(initiate,self.target_angle.abs(),self.prev_error))
        self.u.copy_(self.base10.expand_as(self.u))
        man=torch.where(self.started[:,None],raw,torch.zeros_like(raw))
        self.mlast.mul_(.65).add_(man,alpha=.35);self.u.add_(self.mscale*self.mlast)
        self.u[:,:5].clamp_(0,25);self.u[:,:5].mul_(((self.mlast[:,:5]+1.)/.15).clamp(0.,1.));self.u[:,5:9].clamp_(-22,22);self.u[:,9].clamp_(-95,65)
        release=self.k==25;self.release_center.copy_(torch.where(release[:,None],self.relative(),self.release_center));self.eq[:,self.eqid].logical_and_(~release)
        for _ in range(SUB):wp.capture_launch(self.graph)
        self.k.add_(1);self._contacts()
        self.max_self.copy_(torch.maximum(self.max_self,self.self_pen));self.max_obj.copy_(torch.maximum(self.max_obj,self.object_pen))
        z=self.q[:,self.obj_q+2];rel=self.relative();dist=(rel-self.release_center).norm(dim=1)
        n=(self.force_t>.005).sum(1).int();thumb=(self.force_t[:,-1]>.005).float()
        initial=.25-self.center[:,2];bad=~torch.isfinite(self.q).all(1)|~torch.isfinite(self.v).all(1)
        dropped=(self.k>30)&((z<.09)|(dist>.12))
        _,error=qerror(self.target,self.quat());omega=self.v[:,self.obj_v+3:self.obj_v+6].norm(dim=1)
        active=self.started
        self.opp_count.add_((active&self.opposed).float());self.active_count.add_(active.float())
        self.best_error.copy_(torch.where(active,torch.minimum(self.best_error,error),self.best_error))
        self.lifted_min.copy_(torch.where(active,torch.minimum(self.lifted_min,z),self.lifted_min))
        self.maxdrift.copy_(torch.where(active,torch.maximum(self.maxdrift,dist),self.maxdrift))
        self.hold_contacts.copy_(torch.where(active,torch.minimum(self.hold_contacts,n),self.hold_contacts))
        near=(error<math.radians(8.))&(omega<1.2)&(z>initial+.035)&(n>=2)&(dist<.025)&~self.eq[:,self.eqid]&self.opposed
        self.good_streak.copy_(torch.where(active&near,self.good_streak+1,torch.zeros_like(self.good_streak)))
        excess=(self.total_t-20).clamp(min=0)
        pre=.025*n+.025*thumb-.003*excess
        retain=.5*((z-(initial-.035))/.06).clamp(0,1)+.45*torch.exp(-dist/.025)+.10*n.clamp(max=4)/4
        progress=(self.prev_error-error).clamp(-.2,.2)
        post=retain+.65*self.opposed.float()+.2*(-self.opp_cos).clamp(0,1)+2.2*torch.exp(-error/.28)+3.0*progress+.3*near.float()-.04*omega.clamp(max=15)-.006*excess-.02*(man**2).mean(1)
        reward=torch.where(active,post,pre)-.5*(self.self_pen/.001).clamp(max=10)-.2*(self.object_pen/.002).clamp(max=10);self.prev_error.copy_(torch.where(active,error,self.prev_error))
        done=bad|dropped|(self.k>=self.horizon)
        retained=self.grasped&~bad&~dropped&(self.k>=self.horizon)&(self.lifted_min>initial+.035)&(self.maxdrift<.025)&(self.hold_contacts>=2)&~self.eq[:,self.eqid]
        success=retained&(self.good_streak>=10)&(self.opp_count/self.active_count.clamp(min=1)>=.80)&(self.max_self<.001)&(self.max_obj<.002)
        reward+=torch.where(done,torch.where(success,12.,torch.where(retained,1.,-8.)),0.)
        reward=torch.nan_to_num(reward,nan=-8.,posinf=-8.,neginf=-8.);self.returns.add_(reward)
        info=dict(opposed=self.opposed.clone(),oppositionFraction=(self.opp_count/self.active_count.clamp(min=1)).clone(),oppositionCos=self.opp_cos.clone(),maxSelfPen=self.max_self.clone(),maxObjectPen=self.max_obj.clone(),done=done,success=success,retained=retained,grasped=self.grasped.clone(),bad=bad,return_=self.returns.clone(),active=active.clone(),error=error.clone(),bestError=self.best_error.clone(),streak=self.good_streak.clone(),omega=omega.clone(),targetAngle=self.target_angle.clone(),height=z.clone(),drift=self.maxdrift.clone(),holdContacts=self.hold_contacts.clone(),dropped=dropped.clone(),fixture=self.eq[:,self.eqid].clone())
        if autoreset:self.reset(done)
        return self.obs(),reward,done,info

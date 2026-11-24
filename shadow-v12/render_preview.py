"""Local native rendering from a paired held-out episode; not an aggregate test."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,subprocess,math
from pathlib import Path
import mujoco as mj,numpy as np,torch
from benchmark import gym,load
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/bc01/best.pt');p.add_argument('--out',default='runs/bc01/preview.mp4');a=p.parse_args()
torch.set_num_threads(1);policy,ck=load(a.checkpoint)
blind=json.loads(Path('runs/bc01/blind.json').read_text())['results']
# Choose from declared held-out cases; render a representative success rather than a new success statistic.
paired=zip(blind['zero']['trials'],blind['policy']['trials'])
seed=next(x['seed'] for x,y in paired if y['stable20'] and not x['stable20'])
e=gym.make('AdroitHandPen-v1');obs,_=e.reset(seed=seed);h=e.unwrapped
h.model.vis.global_.offwidth=720;h.model.vis.global_.offheight=540
renderer=mj.Renderer(h.model,height=540,width=720)
cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0.,-.20,.24];cam.distance=.53;cam.azimuth=125;cam.elevation=-15
opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.geomgroup[4]=0
out=Path(a.out)
f=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','720x540','-r','25','-i','-','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
streak=count=0
with torch.inference_mode():
    for k in range(200):
        act=policy(torch.tensor(obs[None],dtype=torch.float32)).numpy()[0]
        obs,reward,_,_,info=e.step(act)
        good=bool(info['success']);count+=good;streak=streak+1 if good else 0
        if k%2==0:
            renderer.update_scene(h.data,camera=cam,scene_option=opt);frame=renderer.render()
            f.stdin.write(np.ascontiguousarray(frame).tobytes())
f.stdin.close();assert f.wait()==0
renderer.close();e.close()
print('PREVIEW',out,'seed',seed,'successful steps',count,'final streak',streak,'slow motion x2',flush=True)

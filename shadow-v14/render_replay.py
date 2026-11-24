"""Render recorded qpos only, with official visual meshes and a gridded cylinder."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,subprocess
from pathlib import Path
import numpy as np,mujoco as mj
from visual_model import make_model
p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
m=make_model();d=mj.MjData(m)
m.vis.global_.offwidth=960;m.vis.global_.offheight=720;m.vis.quality.shadowsize=1024
r=mj.Renderer(m,height=720,width=960)
opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.geomgroup[3]=0;opt.sitegroup[4]=0
cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[.32,0,.315];cam.distance=.40;cam.azimuth=168;cam.elevation=-32
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
f=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','25','-i','-','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
for name in a.inputs:
 x=json.loads(Path(name).read_text())
 print('render',name,x['summary']['angle'],'degrees',flush=True)
 for frame in x['frames'][::2]:
    d.qpos[:]=frame['qpos'];d.qvel[:]=frame['qvel'];d.ctrl[:]=frame['ctrl'];mj.mj_forward(m,d)
    r.update_scene(d,camera=cam,scene_option=opt);f.stdin.write(np.ascontiguousarray(r.render()).tobytes())
f.stdin.close();assert f.wait()==0
r.close();print('REPLAY_RENDERED',out,flush=True)

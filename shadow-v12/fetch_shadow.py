"""Download only pinned upstream E3M5 assets. Verify each Git blob SHA."""
import json,hashlib,concurrent.futures as cf
from pathlib import Path
import requests
commit='eea4c7431b63da8278b0ff31638e0b0e941cec0e'
repo='google-deepmind/mujoco_menagerie'
api=f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'
r=requests.get(api,timeout=30);r.raise_for_status();tree=r.json()
assert not tree.get('truncated')
files=[x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith('shadow_hand/')]
assert files and sum(x['size'] for x in files)<100_000_000
base=Path('.assets')
def fetch(x):
    p=base/x['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): b=p.read_bytes()
    else:
        r=requests.get(f'https://raw.githubusercontent.com/{repo}/{commit}/{x["path"]}',timeout=60);r.raise_for_status();b=r.content
    assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==x['sha'],x['path']
    if not p.exists():p.write_bytes(b)
    return dict(path=x['path'],size=len(b),sha=x['sha'],sha256=hashlib.sha256(b).hexdigest())
with cf.ThreadPoolExecutor(max_workers=6) as ex: rows=list(ex.map(fetch,files))
p=base/'shadow_hand'
assert (p/'LICENSE').exists()
Path('runs/shadow-assets.json').write_text(json.dumps(dict(repo=repo,commit=commit,files=rows),indent=2))
print('SHADOW_ASSETS',len(rows),sum(x['size'] for x in rows),flush=True)
import mujoco as mj
for name in ['right_hand.xml','scene_right.xml']:
    if (p/name).exists():
        m=mj.MjModel.from_xml_path(str(p/name));d=mj.MjData(m);mj.mj_forward(m,d)
        print('MODEL',name,'nq',m.nq,'nv',m.nv,'nu',m.nu,'ngeom',m.ngeom,'ntendon',m.ntendon,flush=True)

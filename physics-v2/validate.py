import json
from pathlib import Path
from trial import run
from model import make_model,load
from copy import deepcopy
from statistics import mean
import numpy as np
from concurrent.futures import ProcessPoolExecutor
P=Path('results')
keys=['center','radius','mass','pull','thumb','oppose','friction']
def work(cfg): return run(**cfg)[0]
if __name__=='__main__':
    search=json.loads((P/'search.json').read_text())
    candidates=[{k:r[k] for k in keys} for r in search if r.get('success')]
    cases=[dict(c,duration=6.) for c in candidates]
    hold=[work(c) for c in cases]
    hold.sort(key=lambda r:(r['success'],r['minHoldDigits'],-r['maxHoldDrift']),reverse=True)
    print('6s hold',json.dumps(hold,indent=2))
    best=hold[0]
    assert best['success'], 'no 6s grasp success'
    cfg={k:best[k] for k in keys}
    nominal,frames=run(**cfg,duration=6,record=True)
    negative,negative_frames=run(**dict(cfg,pull=0.,thumb=0.),duration=6,record=True)
    release,release_frames=run(**cfg,duration=6,release_at=4,record=True)
    assert not negative['success']
    assert not nominal['fixture']
    assert release['finalHeight']<.10
    repeat=work(dict(cfg,duration=6))
    assert repeat==nominal, 'nonreproducible'
    # Perturb mass and friction around the chosen nominal design, not an assurance of physical robustness.
    variants=[dict(cfg,mass=cfg['mass']*mass,friction=cfg['friction']*mu,duration=6.) for mass in [.75,1.,1.25,1.5] for mu in [.8,1.,1.2]]
    with ProcessPoolExecutor(max_workers=8) as p: variations=list(p.map(work,variants))
    print('nominal',json.dumps(nominal,indent=2))
    print('negative',negative['finalHeight'],negative['success'],'release',release['finalHeight'],'variations',sum(r['success'] for r in variations),'/',len(variations))
    summary=dict(nominal=nominal,zeroForce=negative,release=release,variants=variations,deterministic=True)
    (P/'validation.json').write_text(json.dumps(summary,indent=2))
    (P/'config.json').write_text(json.dumps(cfg,indent=2))
    (P/'hand.xml').write_text(make_model(**{k:cfg[k] for k in ['center','radius','mass','friction']}))
    (P/'grasp.json').write_text(json.dumps(dict(summary=nominal,frames=frames),separators=(',',':')))
    (P/'zero-force.json').write_text(json.dumps(dict(summary=negative,frames=negative_frames),separators=(',',':')))
    (P/'release.json').write_text(json.dumps(dict(summary=release,frames=release_frames),separators=(',',':')))

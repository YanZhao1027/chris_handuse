import json
from pathlib import Path
p=Path(__file__).parent/'runs'
for f in ['teachers.json','bc.json','bidir01/history.json','assessment/blind.json']:
    x=p/f
    if not x.exists():continue
    d=json.loads(x.read_text())
    if f=='teachers.json':
        print('Search calibration, best candidate per target:')
        for g,rows in d.items():print(g,'success',rows[0]['success'],'retention',rows[0]['retained'])
    elif f=='bc.json':print('BC successful trials',d['byGoal'],'samples',d['samples'])
    elif f.endswith('history.json'):
        r=d[-1];print('PPO update',r['update'],'steps',r['steps'],'steps/s',round(r['stepsPerSecond']))
        for r in d:
            if 'validation' in r:print('validation',r['update'],r['validation'])
    else:
        for r in d['results']:print('Blind',r['difficulty'],r['angle'],r['scores'])

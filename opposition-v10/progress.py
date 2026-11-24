import json
from pathlib import Path
p=Path(__file__).parent/'runs'
for n in ['mount-calibration.json','mount-holdout.json','gpu-probe.json','sequence-teachers.json','bc.json','opposition01/history.json','assessment/blind.json']:
 f=p/n
 if not f.exists():continue
 d=json.loads(f.read_text())
 if n=='mount-calibration.json':print('Mount calibration',d[0]['success'],'/8, retained',d[0]['retained'],'opposition',d[0]['opposition'])
 elif n=='mount-holdout.json':print('Mount holdout',sum(x['success'] for x in d),'/',len(d),'max self mm',max(x['maxSelfMM'] for x in d),'max object mm',max(x['maxObjectMM'] for x in d))
 elif n=='gpu-probe.json':print('GPU probe',d)
 elif n=='sequence-teachers.json':
  for g,r in d.items():print('Teacher',g,r[0]['success'],r[0]['retained'])
 elif n=='bc.json':print('BC',d['byGoal'],d['samples'])
 elif n.endswith('history.json'):
  print('PPO',d[-1]['update'],'steps',d[-1]['steps'],'rate',round(d[-1]['stepsPerSecond']))
  for row in d:
   if 'validation' in row:print('Validation',row['update'],row['validation'])
 else:
  for row in d['results']:print('Blind',row['difficulty'],row['angle'],row['scores'])
